from verify_forward import run_original_model_forward
from differentiable_forward import DifferentiableModel
import attack
import os

import sys
import numpy as np
from PIL import Image
import torch

if __name__ == "__main__":
    model = DifferentiableModel()
    num_imgs = 1000

    for fpath in ["cifar10_test", "mnist_test", "imagenet_test"]:
        if os.path.exists(fpath):
            break
    else:
        raise

    ds = []
    for root,_,fs in os.walk(fpath):
        for f in sorted(fs):
            if f == 'labels.txt': continue
            ds.append(np.array(Image.open(os.path.join(root,f))))
    ds = np.stack(ds)/255.0
    if len(ds.shape) == 4:
        ds = ds.transpose((0, 3, 1, 2))
    ds = np.array(ds, dtype=np.float32)

    labels = np.array(list(map(int,open(os.path.join(fpath, "labels.txt")).read().split())))
    
    labels_orig = np.array(run_original_model_forward(fpath))

    adv = attack.attack(model, ds, labels)

    from checks.utils import DISTORTION
    if np.any(np.abs(ds - adv) > DISTORTION*1.01):
        print("Distortion bound of", DISTORTION,
              "exceeded: np.max(np.abs(ds - av)) =",
              np.max(np.abs(ds - adv)))
        exit(0)
    
    labels_new = []
    batch_size = 32
    for i in range(0, len(adv), batch_size):
        batch = adv[i:i + batch_size]
        batch_output = model(torch.tensor(batch)).cpu().detach().numpy()
        labels_new.append(batch_output.argmax(1))
    labels_new = np.concatenate(labels_new)

    clean_acc = (labels_orig == labels).mean()
    adv_acc = ((labels_new == labels) | (labels_new == -1)).mean()
    
    print('clean accuracy', clean_acc)
    print('adversarial accuracy', adv_acc)

    if adv_acc < 2*clean_acc/3:
        np.save("/tmp/adv.npy", adv)
        print("PASSES")
