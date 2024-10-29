from os.path import join as ospj
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils


def print_network(network, name):
    num_params = 0
    if isinstance(network, nn.Module):
        for p in network.parameters():
            num_params += p.numel()
    else:  # assume network is a nn.Parameter object
        num_params = network.numel()
    num_params_m = num_params / 1e6
    print("Number of parameters of %s: %.2fM" % (name, num_params_m))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    x = torch.rot90(x, 3, [2, 3]) # rotate 90 degrees for kidney images
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, x_src, y_src, y_trg, filename):
    print('Translating {} images from domain[{}] to domain[{}].'.format(x_src.size(0), y_src, y_trg))
    N, C, H, W = x_src.size()
    s_trg = nets.style_vectors[y_trg]
    x_fake = nets.generator(x_src, s_trg)
    s_src = nets.style_vectors[y_src]
    x_rec = nets.generator(x_fake, s_src)
    x_concat = [x_src, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_target(nets, x_src, trg_domains_label, filename):
    N, C, H, W = x_src.size()
    x_src_without_bg = [x_src]

    s_trg = nets.style_vectors[trg_domains_label]
    if s_trg.dim() == 2:
        s_trg_list = s_trg.unsqueeze(1).repeat(1, N, 1)
    elif s_trg.dim() == 3:
        s_trg_list = s_trg.unsqueeze(1).repeat(1, N, 1, 1)
    elif s_trg.dim() == 4:
        s_trg_list = s_trg.unsqueeze(1).repeat(1, N, 1, 1, 1)
    else:
        raise NotImplementedError

    x_concat = x_src_without_bg
    for i, s_trg in enumerate(s_trg_list):
        x_fake = nets.generator(x_src, s_trg)
        x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    # save_image(x_concat, N+1, filename)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    num_domains = args.num_domains
    src_domain_label = y_src[0]
    all_domains = list(range(num_domains))
    trg_domains_label = [label for label in all_domains if label != src_domain_label]
    extended_list = random.choices(trg_domains_label, k=x_src.size(0))

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, x_src, y_src, extended_list, filename)

    # target-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_target.jpg' % (step))
    translate_using_target(nets, x_src, trg_domains_label, filename)