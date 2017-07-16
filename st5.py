"""
st5.py

This file contains the graph structure for simpletrain5.
"""

import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.api import keras
from tensorflow.contrib.keras.python.keras.applications import InceptionV3
from tensorflow.contrib.keras.python.keras.engine import Input

import FLAGS
import PARAMS
import re
import st5input


# Global constants describing the MSHAPES data set.
IMAGE_SIZE = PARAMS.IMAGE_SIZE
NUM_CLASSES = PARAMS.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = PARAMS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = PARAMS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
TOWER_NAME = PARAMS.TOWER_NAME

# Constants describing the training process.
MOVING_AVERAGE_DECAY = PARAMS.MOVING_AVERAGE_DECAY  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = PARAMS.NUM_EPOCHS_PER_DECAY  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = PARAMS.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.
INITIAL_LEARNING_RATE = PARAMS.LEARNING_RATE_DECAY_FACTOR  # Initial learning rate.


def inputs(eval_data):
    """Construct input for MSHAPES evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, '')
    (filequeue, (images, labels)) = st5input.inputs(eval_data=eval_data,
                                                                          data_dir=data_dir,
                                                                          batch_size=FLAGS.batch_size)

    # print("Reenqueues: ")
    # print(reenqueues)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return filequeue, images, labels



def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                 global_step,
    #                                 decay_steps,
    #                                 LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)
    # tf.summary.scalar('learning_rate', lr)

    train_op = tf.train.AdamOptimizer().minimize(total_loss)

    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    #     opt = tf.train.GradientDescentOptimizer(lr)
    #     grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    return train_op


# def inference_pretrained(images):
#     this could also be the output a different Keras model or layer
#     input_tensor = Input(shape=(150, 150, 3))  # this assumes K.image_data_format() == 'channels_last'
#
#     model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
#
#     return model

def input_process(name, images):
    """
    Model to extract features from one of the input image. Two layers of convolution and pool
    :param name: name of the input
    :param input_image: tensor_shape = [batch_size, width, height, 3]
    :return: feature logits
    """
    channel_num = images.get_shape().as_list()[3]
    # conv1
    with tf.variable_scope(name):
        with tf.variable_scope('conv') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, 16],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [16], tf.constant_initializer(1e-2))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, 16, 16],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    return pool2


def rotation_invariant_net(name, images):
    """
    A convolutional neural network which maintain the rotation invariance of the input image.
    Reference: "Learning rotation invariant convolutional filters for texture classification" by Diego Marcos, etc
        https://arxiv.org/pdf/1604.06720.pdf
    :param name: the name of network
    :param images: input tensor with shape as [batch_size, 100, 100, 3]
    :return: rotation invariant features
    """

    ROTATION_GROUP_NUMBER = 8
    DISCRETE_ORIENTATION_NUMBER = 16

    with tf.variable_scope(name):
        # a common convolution operation
        with tf.variable_scope('canonical_conv') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[31, 31, 3, ROTATION_GROUP_NUMBER],  # the size of the kernel is larger than those are typically used
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
            biases = _variable_on_cpu('biases', [ROTATION_GROUP_NUMBER], tf.constant_initializer(1e-2))
            canonical_conv = tf.nn.bias_add(conv, biases)

        # rotate each channel for DISCRETE_ORIENTATION_NUMBER times and form ROTATION_GROUP_NUMBER groups
        with tf.variable_scope('oriented_max_pool') as scope:
            arr = []
            ROTATE_ANGLE = 2 * np.pi / float(DISCRETE_ORIENTATION_NUMBER)
            for m in xrange(ROTATION_GROUP_NUMBER):
                group_canonical = canonical_conv[:, :, :, m:m + 1]
                for r in xrange(DISCRETE_ORIENTATION_NUMBER):
                    rot = tf.contrib.image.rotate(group_canonical,
                                                  r * ROTATE_ANGLE,
                                                  'BILINEAR')
                    arr.append(rot)
            concat = tf.concat(arr, axis=3)
            # shape: [batch_size, width, height, ROTATION_GROUP_NUMBER]
            with tf.device('/cpu:0'):
                oriented_max_pool = tf.nn.max_pool(concat,
                                                   ksize=[1,1,1,DISCRETE_ORIENTATION_NUMBER],
                                                   strides=[1,1,1,DISCRETE_ORIENTATION_NUMBER],
                                                   padding='SAME')

        with tf.variable_scope('spatial_max_pool') as scope:
            spatial_max_pool = tf.nn.max_pool(oriented_max_pool,
                                              ksize=[1,2,2,1],
                                              strides=[1,2,2,1],
                                              padding='SAME')
        return spatial_max_pool


def input_inference(name, images):
    rotation_invariant = rotation_invariant_net(name, images)
    return input_process(name, rotation_invariant)


def inference(images):
    """
    Build the model in which firstly extract features from both input images first. Then concat them together

    :param images: Images reterned from distored_inputs() or inputs(), tensor_shape = [batch_size, width, height, 6]
    :return: Logits
    """
    with tf.variable_scope('input') as scope:
        input_feature_L = input_inference('input_L', images[:,:,:,:3])
        input_feature_K = input_inference('input_K', images[:,:,:,3:])
        sh = images.get_shape().as_list()
        input_concat = tf.concat([input_feature_L, input_feature_K], axis=len(sh)-1)

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(input_concat, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


# Old code, not used
# def inference_old(images):
#     """Build the CIFAR-10 model.
#     Args:
#       images: Images returned from distorted_inputs() or inputs().
#     Returns:
#       Logits.
#     """
#     # We instantiate all variables using tf.get_variable() instead of
#     # tf.Variable() in order to share variables across multiple GPU training runs.
#     # If we only ran this model on a single GPU, we could simplify this function
#     # by replacing all instances of tf.get_variable() with tf.Variable().
#     #
#     # conv1
#     with tf.variable_scope('conv1') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[5, 5, 6, 64],
#                                              stddev=5e-3,
#                                              wd=0.0)
#         conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(1e-2))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv1)
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool1')
#     # norm1
#     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm1')
#
#     # conv2
#     with tf.variable_scope('conv2') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[5, 5, 64, 64],
#                                              stddev=5e-2,
#                                              wd=0.0)
#         conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv2)
#
#     # norm2
#     norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm2')
#     # pool2
#     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#
#     # local3
#     with tf.variable_scope('local3') as scope:
#         # Move everything into depth so we can perform a single matrix multiply.
#         reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
#         dim = reshape.get_shape()[1].value
#         weights = _variable_with_weight_decay('weights', shape=[dim, 192],  # 384
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#         _activation_summary(local3)
#
#     # local4
#     with tf.variable_scope('local4') as scope:
#         weights = _variable_with_weight_decay('weights', shape=[192, 86],  # 192
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [86], tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#         _activation_summary(local4)
#
#     # linear layer(WX + b),
#     # We don't apply softmax here because
#     # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
#     # and performs the softmax internally for efficiency.
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = _variable_with_weight_decay('weights', [86, NUM_CLASSES],
#                                               stddev=1 / 86.0, wd=0.0)
#         biases = _variable_on_cpu('biases', [NUM_CLASSES],
#                                   tf.constant_initializer(0.0))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#         _activation_summary(softmax_linear)
#
#     return softmax_linear



def loss(logits, labels):
    """Calculates the cross-entropy loss.
    Args:
      logits: Logits from inference().
      labels: Labels from inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')

    return cross_entropy_mean + 6 - 6



def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op



def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing


    author: The TensorFlow Authors
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))



def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
