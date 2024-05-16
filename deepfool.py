# Adapted from: https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py

import torch
import numpy as np
from torch.autograd import Variable


def deepfool(model, inputs, y=None, targeted=False, num_classes=10, overshoot=0.02, max_iter=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)
    model.eval()
    adv_examples = []
    for img, label in zip(inputs, y) if targeted else zip(inputs, [None] * len(inputs)):
        img = img.unsqueeze(0)
        f0 = model(img).data.cpu().numpy().flatten()
        I = (np.array(f0)).flatten().argsort()[::-1]
        I = I[0:num_classes]
        orig_label = I[0]
        input_shape = img.data.cpu().numpy().shape
        pert_image = img.data.cpu().numpy().copy()
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)
        loop_i = 0
        x = Variable(torch.tensor(pert_image, dtype=torch.float).to(device), requires_grad=True)
        fs = model(x)
        k_i = orig_label
        while (k_i == orig_label and loop_i < max_iter) if not targeted else (k_i != label and loop_i < max_iter):
            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()
            for k in range(1, num_classes):
                zero_gradients(x)
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)
            pert_image = img.data.cpu().numpy() + (1 + overshoot) * r_tot
            x = Variable(torch.tensor(pert_image, dtype=torch.float).to(device), requires_grad=True)
            fs = model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())
            loop_i += 1
        r_tot = (1 + overshoot) * r_tot
        adv_examples.append(torch.tensor(pert_image[0], dtype=torch.float).to(device))
    adv_examples = torch.stack(adv_examples).to(device)
    return adv_examples


def zero_gradients(x):
    if hasattr(x.grad, 'data'):
        x.grad.data.zero_()
