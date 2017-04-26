import tensorflow as tf

def loadBuiltInData():
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    return mnist


def softmaxRegression():
    mnist = loadBuiltInData()
    # images,labels =  load
    x = tf.placeholder(tf.float32,[None,784])
    y_actual = tf.placeholder(tf.float32,[None,10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y_predict = tf.nn.softmax(tf.matmul(x,W)+b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
            if (i % 100 == 0):
                print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.5)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def nnClassfier():
    mnist = loadBuiltInData()

    x = tf.placeholder(tf.float32,[None,784])
    y_actual = tf.placeholder(tf.float32,[None,10])
    in_size = 784
    out_size = 10
    hidden_layer_size = 90
    hidden_layer = add_layer(x,in_size,hidden_layer_size,tf.nn.relu)

    output_layer = add_layer(hidden_layer,hidden_layer_size,out_size,activation_function=None)
    y_predict = tf.nn.softmax(output_layer)
    # 	Square loss
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_actual - y_predict),
    #                                     reduction_indices=[1]))

    #   Cross entropy loss
    # loss = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_actual,logits=y_predict,name='xentropy')
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x: batch_xs, y_actual: batch_ys})
            if i % 100 == 0:
                print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))



if __name__ == '__main__':
    nnClassfier()
    # softmaxRegression()
