import tensorflow as tf


def read(path):
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer(path)
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(
            serialized_example,
            features={
                    'images':tf.FixedLenFeature([],tf.string),
                    'labels':tf.FixedLenFeature([],tf.string)})
    images=tf.decode_raw(features['images'],tf.uint8)
    labels=tf.decode_raw(features['labels'],tf.int32)
    with tf.Session() as sess:
        images=sess.run(images)
        labels=sess.run(labels)
    return images,labels