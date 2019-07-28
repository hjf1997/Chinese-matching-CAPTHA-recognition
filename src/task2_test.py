# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:49 2018

@author: p0werHu
"""
import scipy.io as sio
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import scipy.io as sio  
import os

def data_preprocess(root,index):
    train_data = np.zeros((1, 60*200))
    img = mpimg.imread(root + '\\' + "%04d" % index + '.jpg')
    if (img.shape == (60,200)):
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
            w_out = tf.get_variable("w_out",[512,C*5])
        with tf.name_scope('bias'):
            b_out = tf.get_variable("b_out",[C*5])
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)

    return out

def vec2text(temp):
    text_1 =  tf.reshape(tf.argmax(temp[:,0:36], 1),[-1,1])
    text_2 =  tf.reshape(tf.argmax(temp[:,36:72], 1),[-1,1])
    text_3 =  tf.reshape(tf.argmax(temp[:,72:108], 1),[-1,1])
    text_4 =  tf.reshape(tf.argmax(temp[:,108:144], 1),[-1,1])
    text_5 =  tf.reshape(tf.argmax(temp[:,144:180], 1),[-1,1])
    text =  tf.concat([text_1,text_2,text_3,text_4,text_5], 1)
    return text

def transform(index):
    if index < 10:
        return str(index)
    elif index == 10: return 'A'
    elif index == 11: return 'B'
    elif index == 12: return 'C'
    elif index == 13: return 'D'
    elif index == 14: return 'E'
    elif index == 15: return 'F'
    elif index == 16: return 'G'
    elif index == 17: return 'H'
    elif index == 18: return 'I'
    elif index == 19: return 'J'
    elif index == 20: return 'K'
    elif index == 21: return 'L'
    elif index == 22: return 'M'
    elif index == 23: return 'N'
    elif index == 24: return 'O'
    elif index == 25: return 'P'
    elif index == 26: return 'Q'
    elif index == 27: return 'R'
    elif index == 28: return 'S'
    elif index == 29: return 'T'
    elif index == 30: return 'U'
    elif index == 31: return 'V'
    elif index == 32: return 'W'
    elif index == 33: return 'X'
    elif index == 34: return 'Y'
    elif index == 35: return 'Z'

def model(IMAGE_HEIGHT, IMAGE_WIDTH, C, keep_prob_value, root, pat):
    
    X,keep_prob = creat_variable(C, IMAGE_HEIGHT, IMAGE_WIDTH)
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob, C)
    #Y = one_hot_matrix(Y_train.reshape(-1), C)
    #print(Y.shape)
    #print(output.shape)
    #print(Y.shape)
    saver = tf.train.Saver()
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            predict = vec2text(output)
    
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pat+"\\model.ckpt")
        string = ''
        for i in range(50000): #此处输入样例数量，编号从0开始    
            test_data = data_preprocess(root, i)
            p = sess.run(predict, feed_dict={X: test_data, keep_prob: keep_prob_value})
            #print(p)
            string += "%04d" % i +',' + transform(p[0,0]) + transform(p[0,1]) + transform(p[0,2]) + transform(p[0,3]) +transform(p[0,4]) +'\n'
        fout = open('mappings.txt','wt')
        fout.write(string)
        fout.close()
        print('end')
    
if __name__ == "__main__":
    
    root = 'E:\\文档\\验证码识别\\处理后的数据\data-2\\'#此处输入题目二的模型地址
    pat = 'E:\\文档\\验证码识别\\model\\task2' #此处输入题目二的图片经过matlab处理后的文件夹地址，见文档
    model(IMAGE_HEIGHT=60, IMAGE_WIDTH=200, C=36, keep_prob_value=0.7, root=root, pat=pat)
