from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

    

def main():
    '''Manually create dummy model and calculate loss funtion'''
    
    # define inputs
    x = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)

    # create model
    W = tf.Variable([.3], dtype=tf.float32)   # actual optimal weight is [-1]
    b = tf.Variable([-.3], dtype=tf.float32)  # actual optimal is [1]
    y = W * x + b
       
    # write loss funtion
    squared_deltas = tf.square(y - y_)
    loss = tf.reduce_sum(squared_deltas)

    # create optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # input data
    feed = { x: [1,  2,  3,  4],
             y_: [0, -1, -2, -3] }
             
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        # just run forward model
        res_y, res_loss = sess.run([y, loss], feed_dict=feed)
        print('res_y:', res_y, 'res_loss:', res_loss)
        
        # train
        for i in range(1000):
            sess.run(train, feed)
            
        res_W, res_b = sess.run([W, b])
        print('res_W:', res_W, 'res_b:', res_b)
        
        # run forward model again
        res_y, res_loss = sess.run([y, loss], feed_dict=feed)
        print('res_y:', res_y, 'res_loss:', res_loss)
        
    
if __name__ == '__main__':
    main()