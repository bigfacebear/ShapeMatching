#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file for simpletrain5
"""

import base64

from datetime import datetime
import tensorflow as tf
import numpy as np

import st5
from utils import *



def train():
    # Make logging very verbose
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, operation_timeout_in_ms=600000)) as sess: # Stop after 10 minutes

        # Get images and labels for MSHAPES
        print("Setting up getting batches and labels")
        filequeue, images_batch, labels_batch = st5.inputs(eval_data=False)
        print("Got two batches")

        print("Image batch shape: ")
        print(images_batch.get_shape())
        print("Labels batch shape:")
        print(labels_batch.get_shape())

        global_step = tf.contrib.framework.get_or_create_global_step()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print("Building graph...")
        logits = st5.inference(images_batch)
        print("Building graph Done.")

        # Calculate loss.
        loss = st5.loss(logits, labels_batch)
        sum_loss = tf.summary.scalar('loss', loss)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = st5.train(loss, global_step)

        print("Actually running now")

        saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        summary_op_merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(st5.FLAGS.train_dir, sess.graph)

        if FLAGS.RESTORE:
            print("Restoring...")

            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.RESTORE_FROM)

            print("Restored.")
        else:
            print("Initializing global variables")
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()
            print("Finished")

        print("Starting coordinator and queue runners")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        print("Ok")

        # summary_merged_op = tf.summary.merge_all()
        # sess.run(summary_merged_op)
        # summary_writer = tf.summary.FileWriter('./summary/', graph=sess.graph)
        # tf.group(enqueues, reenqueues)

        # enqueue everything as needed
        # e = sess.run([enqueues])
        # print("Enqueue result ")
        # print(e)

        # queue_size = sess.run(filequeue.size())
        # print("Initial queue size: " + str(queue_size))

        for i in xrange(0, 2000):

            # if i % 1000 == 0:
            #     summary_str = sess.run(summary_merged_op)
            #     summary_writer.add_summary(summary_str, i)
            # print("blah")
            _, my_loss = sess.run([train_op, loss])
            # sess.run([reenqueues])
            # m = tf.group(train_op, reenqueues)

            # _, my_loss = sess.run([m, loss])

            # queue_size = sess.run(filequeue.size())
            # print("Queue size: " + str(queue_size))

            ml = np.array(my_loss)

            print('Step: %d     Cross entropy loss: % 6.2f' % (i, ml))

            if i % 100 == 0:
                sum_str = sess.run(sum_loss)
                train_writer.add_summary(sum_str, i)

            if i % 100 == 0 and i != 0:  # Every 1000 steps, save the results and send an email
                print("NS")

                notify("Current cross-entropy loss: " + str(ml) + ".", subject="Running stats [step " + str(i) + "]")

                saver.save(sess, "summaries/netstate/saved_state")#, global_step=i)
                summary_str = sess.run(summary_op_merged)
                train_writer.add_summary(summary_str, i)
                # saver.save(sess, "summaries/netstate/saved_state/model.ckpt")

            if np.isnan(ml):
                print("Oops")
                notify("Diverged :(", subject="Process ended")
                sys.exit(0)

            # Get an image tensor and print its value.
            # print("Getting image tensor")
            # image_tensor = sess.run([images_batch])
            # print("Got image tensor")
            # print(image_tensor[0][50, 50, 1])

            # it = np.array(image_tensor)
            # print(np.shape(it))

            # image1 = image_tensor[0][0]

            # it = np.array(image1)
            # print(np.shape(it))

            # w, h = 200, 100
            # data = np.zeros((h, w, 3), dtype=np.uint8)
            # data[50, 50] = [255, 0, 0]
            # img = Image.fromarray(it, 'RGB')
            # img.save('my.png')
            # img.show()

        print("Ok done with that")

        # Finish off the filename queue coordinator.
        print("Requesting thread stop")
        coord.request_stop()
        print("Ok")
        print("Joining threads")
        coord.join(threads)
        print("Ok")

        print("Finished")



def main(argv=None):
    print("Hello, world! v5")

    # notify("Running simple4train/train.py", subject="Hi!!!")

    missing_dependencies = check_dependencies_installed()

    if len(missing_dependencies) > 0:
        raise Exception("Not all dependencies are installed! (Missing packages " + ' and '.join(missing_dependencies) +
                        "). See README.md for details.")

    if get_hostname() == "lebennin-vm":
        print("Привет!")

    # Download and extract the dataset if it's missing (only on Titan)
    print("Setting up dataset...")
    if get_hostname() == "titan" or FLAGS.CHECK_DATASET:
        # maybe_download_and_extract()
        pass

    print("Done.")

    # Run some checks on the dataset to make sure it's correct
    print("Running tests...")
    verify_dataset()
    print("All tests passed.")

    # Clean up directories
    print("Cleaning up directories...")
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    summary_dir = 'summaries/netstate'
    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)
    tf.gfile.MakeDirs(summary_dir)
    print("Done.")

    # Get images and labels for MSHAPES
    # print("Setting up getting batches and labels")
    # enqueues, reenqueues, images_batch, labels_batch = st4.inputs(eval_data=False)
    # print("Got two batches")

    # print("Image batch shape: ")
    # print(images_batch.get_shape())
    # print("Labels batch shape:")
    # print(labels_batch.get_shape())

    # Train the network!!
    train()



if __name__ == '__main__':
    tf.app.run()
