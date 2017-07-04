import tensorflow as tf
import os
import numpy as np


def resnetpreprocessing(x):
    x = tf.to_float(x)
    # RGB -> BGR and subtract mean image, see
    # keras.applications.resnet50
    # https://groups.google.com/forum/#!topic/caffe-users/wmOnKLSKfpI
    # https://arxiv.org/pdf/1512.03385v1.pdf
    # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    return x[:, :, :, ::-1] - np.array([103.939, 116.779, 123.68]).reshape(1, 1, 1, -1)


def read_labeled_image_list(image_dir):
    """
    Returns:
       List with all filenames in file image_list_file
    """
    dir_list = [x for x in os.walk(os.path.join(os.getcwd(), image_dir))][1:]

    filenames = []
    labels = []

    for i, d in enumerate(dir_list):
        for fname in d[2]:
            filename = os.path.join(d[0], fname)
            label = i
            filenames.append(filename)
            labels.append(int(label))

    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    filename = input_queue[0]
    label = input_queue[1]
    file_contents = tf.read_file(filename)
    example = tf.image.decode_image(file_contents, channels=3)
    example.set_shape([None, None, 3])

    return example, label


def read_img(data_dir, batch_size, shuffle):
    # Reads pfathes of images together with their labels
    def input_fn():
        image_list, label_list = read_labeled_image_list(data_dir)

        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    shuffle=shuffle,
                                                    capacity=batch_size * 5,
                                                    name="file_input_queue")

        image, label = read_images_from_disk(input_queue)

        # resize image
        image = tf.image.resize_images(image, (256, 256), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Image and Label Batching
        image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 10,
                                                  num_threads=1, name="batch_queue",
                                                  allow_smaller_final_batch=True)

        return tf.identity(image_batch, name="features"), tf.identity(label_batch, name="label")

    return input_fn
