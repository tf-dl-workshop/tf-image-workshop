from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os
import numpy as np

features = np.load(os.getcwd() + '/embedding.npy')

features_tensor = tf.Variable(features, name="features")

writer = tf.summary.FileWriter(os.getcwd() + '/_emb')
config = projector.ProjectorConfig()

emb_conf = config.embeddings.add()
emb_conf.tensor_name = features_tensor.name
emb_conf.sprite.image_path = os.getcwd() + '/misc/sprite_valid.jpg'
emb_conf.metadata_path = os.getcwd() + '/misc/label_valid.tsv'
emb_conf.sprite.single_image_dim.extend((72, 72))
projector.visualize_embeddings(writer, config)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer.add_graph(sess.graph)
saver = tf.train.Saver()
saver.save(sess, os.path.join(os.getcwd() + '/_emb', "_model.ckpt"), 1)
