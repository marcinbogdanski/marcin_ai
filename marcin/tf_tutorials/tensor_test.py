from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

def main():
    # Declare constants
    # c1 = tf.constant('haha')
    # c2 = tf.constant( [[0, 1], [2, 3]] )
    # c3 = tf.constant( [2.0, 1.0, 0.1] )
        
    p1 = tf.placeholder(dtype=tf.float32)
    # p2 = tf.placeholder(tf.float32, [None, 768])
    
    v1 = tf.Variable(5, dtype=tf.float32)
    # v2 = tf.Variable(tf.truncated_normal([1]))
    
    z = tf.add(p1, v1)
    
    # soft_max = tf.nn.softmax(c3)
    # one_hot = tf.constant( [1.0, 0.0, 0.0] )
    
    # ln = tf.log(soft_max)
    # ml = tf.multiply(ln, one_hot)
    # en = -1 * tf.reduce_sum(ml)
    
    init = tf.global_variables_initializer()
    feed = {p1: 3.5}
    
    with tf.Session() as sess:
        sess.run(init)
        
        # returns numpy array
        res = sess.run(z, feed_dict=feed)
        print(res)
        
        # request multiple values from single run
        res_z, res_v1 = sess.run([z, v1], feed)
        
        # out = sess.run(v2)
        # print('Normal dist.:', out)
        # out = sess.run(soft_max)
        # print('Softmax:', out)
        # out = sess.run(en)
        # print('Cross Entropy:', out)
        # print(type(out))
        
        
def main2():
    c2 = tf.constant( [[0, 1], [2, 3]], dtype=tf.float32 )
    c3 = tf.constant( [[3, 2], [1, 0]], dtype=tf.float32 )

    z = tf.add(c2, c3)
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run(z)
        print(res)
        fw = tf.summary.FileWriter('log', sess.graph)
        

    
if __name__ == '__main__':
    main()
