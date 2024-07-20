import tensorflow as tf

# 创建一个简单的计算图来测试GPU
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print("If GPU is working, you should see a GPU device in the list below:")
print(tf.config.experimental.list_physical_devices('GPU'))
print("Result of matrix multiplication: \n", c)
