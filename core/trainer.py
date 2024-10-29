import os
from os.path import join as ospj
import time
import datetime
import random
from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
from core import utils
from util.show_HTML import save_loss


class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.optims = Munch()
        for net in self.nets.keys():
            params = self.nets[net].parameters() if isinstance(self.nets[net], nn.Module) else [self.nets[net]]
            self.optims[net] = torch.optim.Adam(
                params=params,
                lr=args.lr,
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

        self.to(self.device)

        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if 'ema' not in name:
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.train,)
        # fetch random validation images for debugging
        fetcher_val = InputFetcher(loaders.val)
        inputs_val = next(fetcher_val)

        print('Start training...')
        start_time = time.time()
        for i in range(args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src

            # sample random target domain labels
            num_domains = args.num_domains
            all_domains = list(range(num_domains))
            y_trg = torch.tensor(random.choices(all_domains, k=x_real.size(0))).to(self.device)

            # train the discriminator
            d_loss, d_losses_dict = compute_d_loss(nets, x_real, y_org, y_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator and the style vectors
            g_loss, g_losses_dict = compute_g_loss(nets, args, x_real, y_org, y_trg)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_vectors.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.style_vectors, nets_ema.style_vectors, beta=0.999)

            if (i + 1) % 5 == 0:
                loss_dict = dict(**d_losses_dict, **g_losses_dict)
                save_loss(args.show_dir, loss_dict)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_dict, g_losses_dict], ['D/loss_', 'G/loss_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)


def compute_d_loss(nets, x_real, y_org, y_trg):
    # with real images
    x_real.requires_grad_()

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_vectors[y_trg]
        x_fake = nets.generator(x_real, s_trg)

    loss = nets.discriminator.module.calc_dis_loss(x_fake, x_real, y_trg, y_org)

    return loss, Munch(d_loss=loss.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg):
    # adversarial loss
    s_trg = nets.style_vectors[y_trg]
    x_fake = nets.generator(x_real, s_trg)

    loss_adv = nets.discriminator.module.calc_gen_loss(x_fake, y_trg)

    # identity loss
    loss_idt = torch.tensor(0.)
    if args.lambda_idt > 0:
        # find the positions where the source domain and target domain are the same
        idt_positions = (y_org == y_trg)
        if idt_positions.any():
            x_idt_real = x_real[idt_positions]
            x_idt_fake = x_fake[idt_positions]
            loss_idt = torch.mean(torch.abs(x_idt_fake - x_idt_real))

    # cycle-consistency loss
    s_org = nets.style_vectors[y_org]
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_cyc * loss_cyc + args.lambda_idt * loss_idt

    return loss, Munch(g_fake_adv=loss_adv.item(),
                       cyc=loss_cyc.item(),
                       idt=loss_idt.item())


def moving_average(model, model_test, beta=0.999):
    if isinstance(model, nn.Module):
        params = model.parameters()
        params_test = model_test.parameters()
    else:  # assume model is a nn.Parameter object
        params = [model]
        params_test = [model_test]

    for param, param_test in zip(params, params_test):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
