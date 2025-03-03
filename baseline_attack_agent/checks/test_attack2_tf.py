import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

from checks.verify_forward import run_original_model_forward
from differentiable_forward import DifferentiableModel
import attack

if __name__ == "__main__":
    new_model = DifferentiableModel()
    num_imgs = 1000
    for fpath in ["cifar10_test", "mnist_test", "imagenet_test"]:
        if os.path.exists(fpath):
            break
    else:
        raise
    ds = []
    for root, _, fs in os.walk(fpath):
        for f in sorted(fs):
            if f == 'labels.txt': continue
            ds.append(np.array(Image.open(os.path.join(root, f))))
    ds = np.stack(ds) / 255.0

    labels = np.array(list(map(int,open(os.path.join(fpath, "labels.txt")).read().split())))
    
    labels_orig = np.array(run_original_model_forward(fpath))
    

    """
    # Convert numpy array to TensorFlow tensor
    ds_tf = tf.convert_to_tensor(ds, dtype=tf.float32)

    adv = attack.attack(new_model, ds_tf, tf.convert_to_tensor(labels, dtype=tf.int64))
    
    labels_new = new_model(adv)
    labels_new = labels_new.numpy().argmax(1)
    """

    sess = new_model.get_session()
    # Convert numpy array to TensorFlow tensor
    ds_tf = tf.placeholder(tf.float32, (len(ds), 32, 32, 3))
    
    # Run the model
    labels_new = new_model(ds_tf)
    
    labels_new = sess.run(labels_new, {ds_tf: ds}).argmax(1)

    clean_acc = (labels_orig == labels).mean()
    adv_acc = (labels_new == labels).mean()
    
    print('clean accuracy', clean_acc)
    print('adversarial accuracy', adv_acc)

    print("Try to see if you can improve the attack to do any better.")
    np.save("/tmp/adv.npy", adv.numpy())
