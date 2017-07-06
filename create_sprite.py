import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import pandas as pd
from skimage import io
from utils import *

W = 72
H = 72

sess = tf.InteractiveSession()

# Reads pfathes of images together with their labels
image_list, label_list = read_labeled_image_list('data/test')

images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            shuffle=False,
                                            num_epochs=1,
                                            capacity=len(image_list),
                                            name="file_input_queue")

image, label = read_images_from_disk(input_queue)

# resize image
image = tf.image.resize_images(image, (H, W), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label], batch_size=len(image_list), capacity=len(image_list),
                                          num_threads=1, name="batch_queue",
                                          allow_smaller_final_batch=True)

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

image_pix, label_val = sess.run([image_batch, label_batch])

import matplotlib.pyplot as plt
import numpy as np

d = np.ceil(len(image_pix) ** 0.5).astype("int")
result = np.zeros((d * H, d * W, 3), dtype="uint8")

for idx, p in enumerate(image_pix):
    col = (idx % d) * W
    row = idx // d * H
    result[row:row + H, col:col + W, :] = p

plt.imshow(result)

classes_map = {0: 'env', 1: 'food', 2: 'front', 3: 'menu', 4: 'profile'}

io.imsave(os.getcwd() + '/misc/sprite_valid.jpg', result)
label_df = pd.DataFrame(list(map(lambda x: classes_map[x], label_val)), columns=['label'])
label_df.to_csv(os.getcwd() + '/misc/label_valid.tsv', sep='\t', index=False, header=False)