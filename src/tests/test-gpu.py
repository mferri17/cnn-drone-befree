# Taken from
# https://stackoverflow.com/a/59712404/10866825

import tensorflow as tf 

tf.debugging.set_log_device_placement(True)
print('\n PHYSICAL DEVICES:', tf.config.list_physical_devices('GPU'), '\n')

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

# You must see an output like this one:
# Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0 
# tf.Tensor([[22. 28.] [49. 64.]], shape=(2, 2), dtype=float32)