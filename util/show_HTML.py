import os
import numpy as np
import cv2
import torch


def save_image(save_url, step, img_dict, batch_size, name='step'):
    if not os.path.exists(save_url + "/img"):
        os.makedirs(save_url + "/img")

    index = step[1: step.find('+')]
    with open(os.path.join(save_url, name + '_{0:05d}.html'.format(int(index))), 'a') as v_html:
        for i in range(batch_size):
            for key in img_dict:
                Image = img_dict[key][i]
                if len(Image.shape) == 3:
                    if isinstance(Image, torch.Tensor):
                        Image = Image.permute([1, 2, 0])
                    else:
                        Image = Image.transpose((1 ,2, 0))
                    Image = ((Image + 1) * 127.5)
                else:
                    Image = ((img_dict[key][i]) * 127.5)
                if not isinstance(Image, np.ndarray):
                    Image = Image.clone().numpy()
                cv2.imwrite("{}/img/{}_{}_{}.jpg".format(save_url, key, step, i), Image.astype(np.uint8).squeeze())
                v_html.write("<img src=\"" + "img/{}_{}_{}.jpg".format(key, step, i) + "\">")
            v_html.write("<br><br>\n")


def save_loss(save_url, text_dict):
    if not os.path.exists(save_url):
        os.makedirs(save_url)
    if not os.path.exists(save_url + "//LossName.txt"):
        with open(save_url + '//LossName.txt', 'w') as f:
            for Index, name in enumerate(text_dict.keys()):
                f.write(name if (Index == 0) else (" " + name))

    with open(save_url + "//loss.txt", 'a') as f:
        for name in text_dict.keys():
            f.write(str(text_dict[name]) + " ")
        f.write("\n")