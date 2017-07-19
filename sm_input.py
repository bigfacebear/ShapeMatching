#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from random import randint

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from utils import print_progress_bar
import FLAGS


IMAGE_SIZE = FLAGS.IMAGE_SIZE
NUM_CLASSES = FLAGS.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


def read_input_correct(filename_queue):
    """
    Reads a pair of MSHAPES records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, lock_image = decode_input(filename_queue[0])
    _, key_image = decode_input(filename_queue[1])

    return lock_image, key_image


def read_input_incorrect(filename_queue):
    """
    Reads a pair of MSHAPES records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, wrong_key_image = decode_input(filename_queue[0])

    return wrong_key_image



def decode_input(file_path):
    """
    Decodes an MSHAPES record.

    :param file_path: The filepath of the png
    :return: A duple containing 0 and the decoded image tensor
    """

    # read the whole file
    serialized_record = tf.read_file(file_path)

    # decode everything into uint8
    image = tf.image.decode_png(serialized_record, dtype=tf.uint8)

    # Cast to float32
    image = tf.cast(image, tf.float32)

    # "Crop" the image.
    # This does not actually do anything, since the image remains the same size; however,
    # it has the effect of setting the tensor shape so that it is inferred correctly in later steps.
    # For details, please see https://stackoverflow.com/a/35692452
    # image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    return 0, image



def inputs(eval_data, data_dir, batch_size):
    """
    Constructs the input for MSHAPES.

    :param eval_data: boolean, indicating if we should use the training or the evaluation data set
    :param data_dir: Path to the MSHAPES data directory
    :param batch_size: Number of images per batch

    :return:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 6] size
        labels: Labels. 1D tensor of [batch_size] size.
    """

    if not eval_data:
        index_beg = 1
        index_end = 2 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN  # TODO: First of all, this should go to (at least) 30k.
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        index_beg = 2 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + 1
        index_end = 2 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + 1 + 2 * NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    print('Enqueuing file names...')
    lock_files = [os.path.join(data_dir, 'images/%d_L.png' % i)
                  for i in xrange(index_beg, index_end, 2)]
    key_files_good = [os.path.join(data_dir, 'images/%d_K.png' % i)
                      for i in xrange(index_beg, index_end, 2)]
    key_files_bad = [os.path.join(data_dir, 'images/%d_K.png' % (i + 1))
                     for i in xrange(index_beg, index_end, 2)]

    for q in [lock_files, key_files_good, key_files_bad]:
        for f in q:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

    good_pairs_queue = tf.train.slice_input_producer([lock_files, key_files_good],
                                                     num_epochs=None, shuffle=True)
    bad_pairs_queue = tf.train.slice_input_producer([key_files_bad],
                                                    num_epochs=None, shuffle=True)
    print('Enqueuing file names done.')

    # Get the correct and incorrect examples from files in the filename queue.
    l, k = read_input_correct(good_pairs_queue)
    wk = read_input_incorrect(bad_pairs_queue)

    correct_example = tf.concat([l, k], axis=2)
    wrong_example = tf.concat([l, wk], axis=2)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    correct_or_incorrect = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    fraction_of_correct = tf.constant(0.5)  # The fraction of correct examples in the input set
    correct_label = tf.constant(1)
    incorrect_label = tf.constant(0)
    image = tf.case({tf.less(correct_or_incorrect, fraction_of_correct): lambda:correct_example,
                     tf.greater(correct_or_incorrect, fraction_of_correct): lambda:wrong_example},
                    default=lambda:correct_example, exclusive=True)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 6])
    label = tf.case({tf.less(correct_or_incorrect, fraction_of_correct): lambda:correct_label,
                     tf.greater(correct_or_incorrect, fraction_of_correct): lambda:incorrect_label},
                    default=lambda:tf.constant(1), exclusive=True)

    return _generate_image_and_label_batch(image, label, min_queue_examples, batch_size,shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 6] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 6] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    print("Image dimensions: ", image.get_shape())
    # image = tf.reshape(image, [2 * IMAGE_SIZE, IMAGE_SIZE, 3])

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size)

    # Display the training images in the visualizer.
    # tf.summary.image('images', images)

    print("Images dimensions: ", images.get_shape())

    return images, tf.reshape(label_batch, [batch_size])
