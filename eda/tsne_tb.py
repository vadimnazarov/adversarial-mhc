import numpy as np
import tensorflow as tf
from resMHystic.mhcdata import open_npz
from tensorflow.contrib.tensorboard.plugins import projector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", "--p",  type=str,
                    help="Path to data to visualise")
parser.add_argument("--name", "--n",  type=str,
                    help="What to visualise")


args = parser.parse_args()

data = open_npz(args.path)
dim = data.shape[1]*data.shape[2]
title=args.name

def generate_embeddings(input_data, name):

    sess = tf.Session()
    embedding = tf.Variable(input_data, trainable=False, name='embedding')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./tsne_{}/'.format(name))

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = "../tsne_metadata/alleles.tsv"

    projector.visualize_embeddings(writer, config)

    saver.save(sess, "./tsne_{}/{}_tsne.ckpt".format(name, name), global_step=input_data.shape[0])

    sess.close()


generate_embeddings(data.reshape([-1, dim]), args.name)
