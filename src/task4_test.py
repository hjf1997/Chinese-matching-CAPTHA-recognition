# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:49 2018

@author: p0werHu
"""
import scipy.io as sio
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio  
import os
import math

def data_preprocess(root,index):
    train_data = np.zeros((1, 45*150))
    img = mpimg.imread(root + '\\' + "%04d" % index + '.jpg')
    if (img.shape == (45,150)):
        train_data[0,:] = img.reshape(1,-1)
    else:
        print('图片大小有问题')
    return train_data
    
        
def creat_variable(C,IMAGE_HEIGHT, IMAGE_WIDTH):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name = 'X')
    keep_prob = tf.placeholder(tf.float32) # dropout

    return X,keep_prob
    
def forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob,C):
    
    with tf.name_scope('input_reshape'):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
 
	# 3 conv layer
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            w_c1 = tf.get_variable("w_c1",[3, 3, 1, 16])
        with tf.name_scope('bias'):
            b_c1 = tf.get_variable("b_c1",[16])
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)
    
    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32])
        with tf.name_scope('bias'):
            b_c2 = tf.get_variable("b_c2",[32])
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.name_scope('conv3'):
        with tf.name_scope('weights'):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64])
        with tf.name_scope('bias'):
            b_c3 = tf.get_variable("b_c3",[64])
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    with tf.name_scope('conv4'):
        with tf.name_scope('weights'):
            w_c4 = tf.get_variable("w_c4",[3, 3, 64, 128])
        with tf.name_scope('bias'):
            b_c4 = tf.get_variable("b_c4",[128])
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob)
        tf.summary.histogram('conv4', conv4)
        
    r = conv4.get_shape()
	# Fully connected layer
    with tf.name_scope('FC'):
        with tf.name_scope('weights'):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512])
        with tf.name_scope('bias'):
            b_d = tf.get_variable("b_d",[512])
        dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
    with tf.name_scope('output'):
        with tf.name_scope('weights'):
            w_out = tf.get_variable("w_out",[512,C])
        with tf.name_scope('bias'):
            b_out = tf.get_variable("b_out",[C])
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)

    return out

def model(IMAGE_HEIGHT, IMAGE_WIDTH, C, keep_prob_value, root):
    
    X,keep_prob = creat_variable(C, IMAGE_HEIGHT, IMAGE_WIDTH)
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob, C)

    saver = tf.train.Saver()
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            predict =  tf.argmax(output, 1)
    
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        pat = 'E:\\文档\\验证码识别\\model\\task4'
        saver.restore(sess, pat+"\\model.ckpt")
        string = ''
        for i in range(5000):    #此处输入样例数量，编号从0开始    
            test_data = data_preprocess(root, i)
            p = sess.run(predict, feed_dict={X: test_data, keep_prob: keep_prob_value})
            string += "%04d" % i +',' + str(p[0]) +'\n'
        fout = open('mappings.txt','wt')
        fout.write(string)
        fout.close()
        print('end')
    
if __name__ == "__main__":
    
    root = 'E:\\文档\\验证码识别\\处理后的数据\data-4_raw\\'#此处输入题目四的模型地址
    pat = 'E:\\文档\\验证码识别\\model\\task4'#此处输入题目三的图片经过matlab处理后的文件夹地址，见文档
    model(IMAGE_HEIGHT=45, IMAGE_WIDTH=150, C=4, keep_prob_value=0.7, root=root)
