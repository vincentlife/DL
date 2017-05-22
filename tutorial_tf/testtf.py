import tensorflow as tf
import numpy as np

def test1():
    # 用 NumPy 随机生成 100 个数据
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 构造一个线性模型
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()

    # 启动图 (graph)
    sess = tf.Session()

    sess.run(init)
    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print (step, sess.run(W), sess.run(b))

def test2():
    # 创建一个 常量 op, 返回值 'matrix1' 代表这个 1x2 矩阵.
    matrix1 = tf.constant([[3., 3.]])

    # 创建另外一个 常量 op, 返回值 'matrix2' 代表这个 2x1 矩阵.
    matrix2 = tf.constant([[2.], [2.]])

    # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
    # 返回值 'product' 代表矩阵乘法的结果.
    product = tf.matmul(matrix1, matrix2)
    sess = tf.Session()
    print(sess.run(product))

def test3():
    # －创建一个变量, 初始化为标量 0.  初始化定义初值
    state = tf.Variable(0, name="counter")

    # 创建一个 op, 其作用是使 state 增加 1
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    init_op = tf.global_variables_initializer()

    # 启动默认图, 运行 op
    with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)

        print(sess.run(state))
        # 也可以在运行一次 op 时一起取回多个 tensor:
        # result = sess.run([mul, intermed])
        print(sess.run([state,new_value]))

        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

def test_gpu():
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            matrix1 = tf.constant([[3., 3.]])
            matrix2 = tf.constant([[2.], [2.]])
            product = tf.matmul(matrix1, matrix2)
            result = sess.run([product])
            print(result)


if __name__ == '__main__':
    test_gpu()