#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file for MSHAPES train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import sm
from utils import *


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = sm.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = sm.inference(images)

        # Calculate loss.
        loss = sm.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = sm.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

        print('Finished.')


def main(argv=None):
    print("Hello, world! v5")

    # notify("Running simple4train/train.py", subject="Hi!!!")

    missing_dependencies = check_dependencies_installed()

    if len(missing_dependencies) > 0:
        raise Exception("Not all dependencies are installed! (Missing packages " + ' and '.join(missing_dependencies) +
                        "). See README.md for details.")

    # if get_hostname() == "lebennin-vm":
    #     print("Привет!")
    #
    # # Download and extract the dataset if it's missing (only on Titan)
    # print("Setting up dataset...")
    # if get_hostname() == "titan" or FLAGS.CHECK_DATASET:
    #     # maybe_download_and_extract()
    #     pass
    #
    # print("Done.")

    # Run some checks on the dataset to make sure it's correct
    # print("Running tests...")
    # verify_dataset()
    # print("All tests passed.")

    # Clean up directories
    print("Cleaning up directories...")
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    print("Done.")

    # Train the network!!
    print('Begin training...')
    train()



if __name__ == '__main__':
    tf.app.run()
