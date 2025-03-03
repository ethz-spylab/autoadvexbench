import tensorflow as tf
import traceback
import numpy as np
from differentiable_forward import DifferentiableModel
from checks.utils import SHAPE_TF

def test_gradient_nonzero_eager(model):
    x = tf.random.uniform((1,)+SHAPE_TF, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x)
        loss = tf.reduce_sum(tf.square(logits))
    
        gradients = tape.gradient(loss, x)
    
    assert gradients is not None, "FAILS: Gradients should not be None"
    assert tf.reduce_sum(tf.abs(gradients)).numpy() != 0, "FAILS: Gradients should not be zero"
    print("PASSES")

if __name__ == "__main__":
    # Test in eager mode
    print("Testing in eager mode:")
    model = DifferentiableModel()
    try:
        tf.compat.v1.enable_eager_execution()
    except:
        pass
    test_gradient_nonzero_eager(model)
