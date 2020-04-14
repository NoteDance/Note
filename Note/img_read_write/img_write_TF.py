import cv2 as cv
import tensorflow as tf
import glob


def write(image_filename,labels,image_sum,path):
    writer=tf.python_io.TFRecordWriter(path)
    if image_sum>1:
        images_filenames=glob.glob(image_filename+'*.jpg')
    if image_sum==1:
        image_array=cv.imread(image_filename)
        image=image_array.tostring()
        feature={'images':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                 'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tostring()]))}
        features=tf.train.Features(feature=feature)
        example=tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        writer.close()
        return
    for i in range(image_sum):
        image_array=cv.imread(images_filenames[i])
        image=image_array.tostring()
        feature={'images':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                 'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels[i].tostring()]))}
    features=tf.train.Features(feature=feature)
    example=tf.train.Example(features=features)
    writer.write(example.SerializeToString())
    writer.close()
    return