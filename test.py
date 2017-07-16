import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# print("Hello")
#
# with tf.Session() as sess:
#     serialized_record = tf.read_file('./MSHAPES/images/100_L.png')
#
#     image = tf.image.decode_png(serialized_record, dtype=tf.uint8)
#     image = tf.cast(image, tf.float32)
#
#     s = tf.shape(image)
#
#     print(sess.run(s))

def testDeconcat():
    i1 = np.zeros([5,4,4,3])
    i2 = np.ones([5,4,4,3])
    t1 = tf.Variable(i1)
    t2 = tf.Variable(i2)
    c = tf.concat([t1, t2], axis=3)

    sh = c.get_shape().as_list()
    t1.get_shape()
    print sh

    dc1 = tf.slice(c, begin=tf.constant(sh[:3]+[0]), size=sh[:3]+[3])
    dc2 = tf.slice(c, begin=tf.constant(sh[:3]+[3]), size=sh[:3]+[3])
    print dc1
    print dc2

    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print sess.run(dc1)

def testImshow():
    ROTATION_GROUP_NUMBER = 8
    DISCRETE_ORIENTATION_NUMBER = 16
    mat = np.zeros([2, 100, 100, ROTATION_GROUP_NUMBER], dtype=np.uint8)
    mat[:, 25:75, 25:75, :] = 255 * np.ones([2, 50, 50, ROTATION_GROUP_NUMBER])

    with tf.variable_scope('rotation'):
        canonical_conv = tf.Variable(mat)
        arr = []
        ROTATE_ANGLE = 2 * np.pi / float(DISCRETE_ORIENTATION_NUMBER)
        for m in xrange(ROTATION_GROUP_NUMBER):
            group_canonical = canonical_conv[:,:,:,m:m+1]
            print group_canonical.get_shape()
            for r in xrange(DISCRETE_ORIENTATION_NUMBER):
                rot = tf.contrib.image.rotate(group_canonical,
                                              r * ROTATE_ANGLE,
                                              'BILINEAR')
                arr.append(rot)
                tf.summary.scalar('rotation', 1)
        concat = tf.concat(arr, axis=3)
        print concat.get_shape()
        rotation = tf.nn.max_pool(concat, ksize=[1,1,1,DISCRETE_ORIENTATION_NUMBER], strides=[1,1,1,DISCRETE_ORIENTATION_NUMBER], padding='SAME')


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    sess.run(merged_summary_op)
    summary_writer = tf.summary.FileWriter('./testSummary/', graph=sess.graph)
    for i in xrange(100):
        out = sess.run(rotation)
        print str(i)+':', out.shape
        if i % 20 == 0:
            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, i)

    # plt.figure('test')
    # plt.imshow(out)
    # plt.show()

testImshow()