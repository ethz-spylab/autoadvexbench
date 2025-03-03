import checks.verify_forward
import numpy as np
from PIL import Image
import os
from differentiable_forward import DifferentiableModel
import attack
import torch

if __name__ == "__main__":
    model = DifferentiableModel()

    ds = []
    for root,_,fs in os.walk("/mnt/dataset/cifar10_test"):
        for f in sorted(fs):
            if f == 'labels.txt': continue
            ds.append(np.array(Image.open(os.path.join(root,f))))
    ds = np.stack(ds).transpose((0, 3, 1, 2))/255.0
    ds = np.array(ds, dtype=np.float32)
    labels = open(os.path.join("/mnt/dataset/cifar10_test", "labels.txt")).read()
    labels = np.array([int(x) for x in labels.split()])

    adv = []
    for i in range(0, len(ds), 32):
        adv.append(attack.attack(model, torch.tensor(ds[i:i+32]), torch.tensor(labels[i:i+32])).detach().cpu().numpy())

    adv = np.concatenate(adv)

    os.mkdir("/tmp/adv")
    for i,x in enumerate(adv):
        Image.fromarray(np.array(x.transpose((1, 2, 0))*255.0, dtype=np.uint8)).save("/tmp/adv/%04d.png"%i)
