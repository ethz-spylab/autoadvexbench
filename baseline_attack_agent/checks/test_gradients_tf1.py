import tensorflow as tf
import traceback
import numpy as np
from differentiable_forward import DifferentiableModel
from checks.utils import SHAPE_TF

def test_gradient_nonzero_v1(model):
    try:
        tf.compat.v1.disable_eager_execution()
    except:
        pass
    
    x = tf.placeholder(tf.float32, shape=(1,)+SHAPE_TF)
    logits = model(x)
    loss = tf.reduce_sum(tf.square(logits))
    
    gradients = tf.gradients(loss, x)[0]
    
    sess = model.get_session()
    grad_value = sess.run(gradients, feed_dict={x: np.array(np.random.randn(1, *SHAPE_TF), dtype=np.float32)})
    
    assert grad_value is not None, "FAILS: Gradients should not be None"
    assert np.sum(np.abs(grad_value)) != 0, "FAILS: Gradients should not be zero"
    print("PASSES")

if __name__ == "__main__":
    # Test in eager mode
    model = DifferentiableModel()
    
    test_gradient_nonzero_v1(model)
