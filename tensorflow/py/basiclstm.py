# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:13:46 2018

@author: omf
"""

import tensorflow as tf  
  
batch_size = 4  
#step batch_size,?
input = tf.random_normal(shape=[4,batch_size, 30], dtype=tf.float32)   
cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)  
init_state = cell.zero_state(batch_size, dtype=tf.float32)  
output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, time_major=True) 
#time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。  
#如果是True，output的维度是[max_time, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
print(output)
print(cell,'\n')
print(final_state)
'''
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    print(sess.run(output))  
    print(sess.run(final_state)) '''