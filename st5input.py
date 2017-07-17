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
from PARAMS import *


# The image size to crop the image to.
# Keep at 100 to avoid any cropping;
# do not set this value below 80
# IMAGE_SIZE = DIM



def read_mshapes_correct(filename_queue):
    """
    Reads a pair of MSHAPE records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, lock_image = decode_mshapes(filename_queue[0])
    _, key_image = decode_mshapes(filename_queue[1])

    # Combine images to make a correct example and an incorrect example
    # correct_example = tf.concat([lock_image, key_image], axis=0)

    # print("Correct example", correct_example)

    # Return the examples
    return lock_image, key_image


def read_mshapes_incorrect(filename_queue):
    """
    Reads a pair of MSHAPE records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, wrong_key_image = decode_mshapes(filename_queue[0])

    return wrong_key_image



def decode_mshapes(file_path):
    """
    Decodes an MSHAPE record.

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
    # This does not actually do anything, since
    # the image remains the same size; however,
    # it has the effect of setting the tensor
    # shape so that it is inferred correctly
    # in later steps. For details, please
    # see https://stackoverflow.com/a/35692452
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

    # filequeue = tf.FIFOQueue(capacity=100000, dtypes=[tf.string, tf.string])  # FIXME: Use RandomShuffleQueue instead!!
    # enqueues = []

    lock_files = []
    key_files_good = []
    key_files_bad = []

    if not eval_data:
        print("Not eval data")

        print_progress_bar(0, NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete', length=50,
                           fill='█')

        for i in xrange(1, NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, 2):  # TODO: First of all, this should go to (at least) 30k.
            # The reason it's at 5000 is that currently, we're
            # individually enqueueing images. Instead, we should
            # use enqueue_many with an inline for loop, which
            # should building up the queue much faster.

            print_progress_bar(i + 1, NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete', length=50,
                               fill='█')

            lock = os.path.join(data_dir, 'images/%d_L.png' % i)
            key_good = os.path.join(data_dir, 'images/%d_K.png' % i)
            key_bad = os.path.join(data_dir, 'images/%d_K.png' % (i + 1))

            lock_files.append(lock)
            key_files_good.append(key_good)
            key_files_bad.append(key_bad)

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print("Ok")
    else:
        # for i in xrange(30001, 49500): #FIXME Same thing as above
        #     lock = os.path.join(data_dir, 'images/%d_L.png' % i)
        #     key = os.path.join(data_dir, 'images/%d_K.png' % i)
        #
        #     a = filequeue.enqueue([lock, key])
        #     enqueues.append(a)

        beg = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        end = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        for i in xrange(beg, end, 2):  # TODO: First of all, this should go to (at least) 30k.
            # The reason it's at 5000 is that currently, we're
            # individually enqueueing images. Instead, we should
            # use enqueue_many with an inline for loop, which
            # should building up the queue much faster.

            # print_progress_bar(i + 1, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, prefix='Progress:', suffix='Complete', length=50,
            #                    fill='█')

            lock = os.path.join(data_dir, 'images/%d_L.png' % i)
            key_good = os.path.join(data_dir, 'images/%d_K.png' % i)
            key_bad = os.path.join(data_dir, 'images/%d_K.png' % (i + 1))

            lock_files.append(lock)
            key_files_good.append(key_good)
            key_files_bad.append(key_bad)

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        print("Ok")


    # print("Lock files: ")
    # print(lock_files)
    # print("Good key files: ")
    # print(key_files_good)
    # print("Bad key files: ")
    # print(key_files_bad)

    good_pairs_queue = tf.train.slice_input_producer([lock_files, key_files_good],
                                              num_epochs=None, shuffle=True)
    bad_pairs_queue = tf.train.slice_input_producer([key_files_bad],
                                                     num_epochs=None, shuffle=True)

    print("Finished enqueueing")

    # Get the correct and incorrect examples from files in the filename queue.
    l, k = read_mshapes_correct(good_pairs_queue)
    wk = read_mshapes_incorrect(bad_pairs_queue)

    correct_example = tf.concat([l, k], axis=2)
    wrong_example = tf.concat([l, wk], axis=2)
    # wrong_example = tf.concat([l, k], axis=2)  # Give it the same wrong example as correct; should get stuck at 0.69 (it does)

    print("c Example shape:--------------------------------------------------------->", correct_example.get_shape())
    print("w Example shape:--------------------------------------------------------->", wrong_example.get_shape())

    print("Got examples")

    # Ensure that the random shuffling has good mixing properties.
    print("Mixing properties stuff")
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Regroup the enqueues
    # grouped_enqueues = tf.group(enqueues[0], enqueues[1])
    # for i in xrange(2, len(enqueues) - 1):
    #     grouped_enqueues = tf.group(grouped_enqueues, enqueues[i])

    correct_or_incorrect = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    # The case code is basically tensorflow language for this:
    #
    # if (correct_or_incorrect < 0.5):
    #     _generate_image_and_label_batch(correct_example, [1],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)
    # else:
    #     _generate_image_and_label_batch(wrong_example, [0],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)

    def f1(): return correct_example
    def f2(): return wrong_example

    def g1(): return tf.constant(0)
    def g2(): return tf.constant(1)


    image = tf.case({tf.less(correct_or_incorrect, tf.constant(0.5)): f1, tf.greater(correct_or_incorrect, tf.constant(0.5)): f2},
                    default=f1, exclusive=True)
    # image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 6])
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 6])
    label = tf.case({tf.less(correct_or_incorrect, tf.constant(0.5)): g1, tf.greater(correct_or_incorrect, tf.constant(0.5)): g2},
                    default=g1, exclusive=True)


    return (good_pairs_queue, (_generate_image_and_label_batch(image, label,
                                                               min_queue_examples, batch_size,
                                                               shuffle=True)))

    # def f1(): return (good_pairs_queue, (_generate_image_and_label_batch(correct_example, [1],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)))
    # def f2(): return (good_pairs_queue, (_generate_image_and_label_batch(wrong_example, [0],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)))
    #
    # r = tf.case([(tf.less(correct_or_incorrect, tf.constant(0.5)), f1)], default=f2)
    #
    # return r



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
