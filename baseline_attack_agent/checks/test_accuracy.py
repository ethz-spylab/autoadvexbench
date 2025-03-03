from checks.verify_forward import run_original_model_forward
from tensor_forward import TensorModel
import os
import sys
import numpy as np
from PIL import Image
import torch

if __name__ == "__main__":
    new_model = TensorModel()
    num_imgs = 1000
    batch_size = 64

    for fpath in ["cifar10_test", "mnist_test", "imagenet_test"]:
        if os.path.exists(fpath):
            break
    else:
        raise

    # Load dataset
    ds = []
    for root, _, fs in os.walk(fpath):
        for f in sorted(fs):
            if f == 'labels.txt': continue
            ds.append(np.array(Image.open(os.path.join(root, f))))
    ds = np.stack(ds) / 255.0
    if len(ds.shape) == 4:
        ds = ds.transpose((0, 3, 1, 2))
    ds_tensor = torch.tensor(ds).float()

    # Get ground truth labels
    labels_groundtruth = run_original_model_forward(fpath)
    
    # Process in batches
    labels_new = []
    for i in range(0, len(ds), batch_size):
        batch = ds_tensor[i:i + batch_size]
        batch_output = new_model(batch).cpu()
        labels_new.append(batch_output.argmax(1))
    
    # Concatenate all batch results
    labels_new = torch.cat(labels_new).numpy()

    # Compare results
    same = (labels_new == labels_groundtruth).mean()
    print("Fraction of inputs that match:", same)
    if same > .99:
        print("PASSES")
    else:
        print("The following images differ",
              ["/tmp/testacc/%05d.png" % i for i in np.where(labels_new != labels_groundtruth)[0]][:10])
