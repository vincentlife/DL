import tensorflow as tf
import pandas




def read_my_file_format(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=True)
    key, record_string = reader.read(filename_queue)
    tensorlist = tf.decode_csv(record_string,record_defaults=[[0]]*785)
    features = tf.stack(tensorlist[1:])
    return features,tensorlist[0]


def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

def write2tfrec(filepath):
    csv = pandas.read_csv(filepath).values
    with tf.python_io.TFRecordWriter("csv.tfrecords") as tf_writer:
        for row in csv:
            features, label = row[1:], row[1]
            example = tf.train.Example()
            example.features.feature["label"].int64_list.value.append(label)
            example.features.feature["features"].float32_list.value.extend(features)
            tf_writer.write(example.SerializeToString())


def readTFRecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'features' : tf.FixedLenFeature([], tf.float32),
                                       })
    img = tf.cast(features['features'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img,label

if __name__ == '__main__':
    csvfile = r"D:\DateSet\digit\train.csv"
    TFRecfile = "csv.tfrecords"
    filename_queue = tf.train.string_input_producer([TFRecfile])
    img,label = readTFRecord(filename_queue)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val,l = sess.run([img_batch,label_batch])
            print(l)
            print(val)
            print("-------------------")