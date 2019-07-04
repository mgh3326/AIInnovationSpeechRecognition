import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
print(tf.__version__)
tf.set_random_seed(0)  # for reproducibility
x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
W2 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

learning_rate = tf.Variable(0.001)

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = W1 * x1_data + W2 * x2_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W1_grad, W2_grad, b_grad = tape.gradient(cost, [W1, W2, b])
    W1.assign_sub(learning_rate * W1_grad)
    W2.assign_sub(learning_rate * W2_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
            i, cost.numpy(), W1.numpy()[0], W2.numpy()[0], b.numpy()[0]))
# Before
x_data = [
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]
]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

learning_rate = tf.Variable(0.001)

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, x_data) + b  # (1, 2) * (2, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], b.numpy()[0]))
# Transpose
x_data = [
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]
]
y_data = [1, 2, 3, 4, 5]
x_data = np.array(x_data, dtype=np.float32).T
y_data = (np.array(y_data)).reshape((5, 1))
np.reshape
W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

learning_rate = tf.Variable(0.001)

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(x_data, W) + b  # (1, 2) * (2, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[1][0], b.numpy()[0]))

# Add One line
x_data = [
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.],
    [0., 2., 0., 4., 0.]
]
y_data = [1, 2, 3, 4, 5]
x_data = np.array(x_data, dtype=np.float32).T
y_data = (np.array(y_data)).reshape((5, 1))
np.reshape
W = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

learning_rate = tf.Variable(0.001)

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(x_data, W) + b  # (5,3) * (3,1) = (5,1)
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[1][0], b.numpy()[0]))
