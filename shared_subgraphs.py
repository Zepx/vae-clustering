import tensorflow as tf
from tensorbayes.layers import constant, placeholder, dense, gaussian_sample
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits
import numpy as np
import sys

# vae subgraphs
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = dense(x, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        qy_logit = dense(h2, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        xy = tf.concat((x, y), 1, name='xy/concat')
        h1 = dense(xy, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        zm = dense(h2, 64, 'zm', reuse=reuse)
        zv = dense(h2, 64, 'zv', tf.nn.softplus, reuse=reuse)
        z = gaussian_sample(zm, zv, 'z')
    return z, zm, zv

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)
