import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

from checks.verify_forward import run_original_model_forward
from tensor_forward import TensorModel

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
    ds = np.array(ds, dtype=np.float32)

    labels_groundtruth = run_original_model_forward(fpath)

    # Create TensorFlow dataset for batch processing
    dataset = tf.data.Dataset.from_tensor_slices(ds).batch(batch_size)

    # Process in batches
    labels_new = []
    for batch in dataset:
        batch_output = new_model(batch)
        labels_new.append(batch_output.numpy())

    # Concatenate all batch results and get predictions
    labels_new = np.concatenate(labels_new)
    labels_new = np.argmax(labels_new, axis=1)

    # Compare results
    same = (labels_new == labels_groundtruth).mean()
    print("Fraction of inputs that match:", same)
    if same > .99:
        print("PASSES")
    else:
        print("The following images differ",
              ["/tmp/testacc/%05d.png" % i for i in np.where(labels_new != labels_groundtruth)[0]][:10])
