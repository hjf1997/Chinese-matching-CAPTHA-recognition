# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:59:34 2018

@author: p0werHu
"""
import scipy.io as sio
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import time
import matplotlib.pyplot as plt
import os
##-------------------data process--------------------------------------------------------------
def picture2string(index, triple_data_path):
    img_path = triple_data_path + str(index) + '.jpg'
    img = mpimg.imread(img_path)
    img = np.lib.pad(img, ((0,0),(0,45-img.shape[1])), 'constant', constant_values=(255,255))
    return img

def data_preprocess(root1, root2, index):
    class_path = 'E:\\文档\\验证码识别\\处理后的数据\\data-5\\'
    match_path = 'E:\\文档\\验证码识别\\处理后的数据\\data-5_match\\'
    data_path = class_path + "%04d" % index + '\\'
    data_match_path = match_path + "%04d" % index + '\\'
    data1 =  picture2string(1, data_path)
    data2 = picture2string(2, data_path)
    data3 = picture2string(3, data_path)
    data4 = picture2string(4, data_path)
    match0 = picture2string(0, data_match_path)
    match1 = picture2string(1, data_match_path)
    match2 = picture2string(2, data_match_path)
    match3 = picture2string(3, data_match_path)
    match4 = picture2string(4, data_match_path)
    match5 = picture2string(5, data_match_path)
    match6 = picture2string(6, data_match_path)
    match7 = picture2string(7, data_match_path)
    match8 = picture2string(8, data_match_path)
    data = []
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(match0)
    data.append(match1)
    data.append(match2)
    data.append(match3)
    data.append(match4)
    data.append(match5)
    data.append(match6)
    data.append(match7)
    data.append(match8)
    return data
    
#--------------------------------------------------------------------------------------------------------
    

#------------------------------------training---------------------------------------------------------

def creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH):
    data = tf.placeholder(tf.float32, [IMAGE_HEIGHT, IMAGE_WIDTH], name = 'data')
    keep_prob = tf.placeholder(tf.float32) # dropout
    
    return data, keep_prob

def forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob):
    
    with tf.variable_scope('input_reshape',reuse=tf.AUTO_REUSE):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
	# 3 conv layer
    x = x / 255. - 0.3
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c1 = tf.get_variable("w_c1",[3, 3, 1, 16])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c1 = tf.get_variable("b_c1",[16])
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)
    
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c2 = tf.get_variable("b_c2",[32])
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c3 = tf.get_variable("b_c3",[64])
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    
    with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c4 = tf.get_variable("w_c4",[3, 3, 64, 128])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c4 = tf.get_variable("b_c4",[128])
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob)
        tf.summary.histogram('conv4', conv4)
        
    r = conv4.get_shape()
	# Fully connected layer
    with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_d = tf.get_variable("b_d",[512])
        dense = tf.reshape(conv4,[-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
    
    with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_out = tf.get_variable("w_out",[512,64])
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_out = tf.get_variable("b_out",[64])
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    return out

def model(IMAGE_HEIGHT, IMAGE_WIDTH, keep_prob_value, minibatch_size, pat, root1, root2):
    
    data,keep_prob = creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, data, keep_prob)

    saver = tf.train.Saver()  
    
    with tf.Session() as sess:
        saver.restore(sess, pat+"\\model.ckpt")
        root1 = ''
        root2 = ''
        string = ''
        for it in range(5000): #此处输入样例数量，编号从0开始    
            data_test = data_preprocess(root1, root2, it)
            
            features = []
            for i in range(0,13):
                
                feature = sess.run(output,feed_dict={data:data_test[i], keep_prob: 1})
                features.append(feature)
            distance = []
            for i in range(4):
                dis = []
                for j in range(4,13,1):
                    dis.append(np.linalg.norm(features[i] - features[j]))
                distance.append(dis)
            dist = np.array(distance)
            predict = 0
            for i in range(4):
                index = np.argmin(dist)
                row = int(index/9)
                col = index%9
                dist[row,:] = 200
                dist[:,col] = 200
                predict += col * 10**(3-row)

            string += "%04d" % it +',' + str("%04d" % predict) + '\n'
        fout = open('mappings.txt','wt')
        fout.write(string)
        fout.close()
        #print(string)
#-----------------------------------------end-----------------------------------------------------
#-------------------------------------Cross-validation---------------------------------------------------


if __name__ == '__main__':
    pat = 'E:\\文档\\验证码识别\\model\\task5' #此处输入题目五的模型文件夹
    root1 = 'E:\\文档\\验证码识别\\处理后的数据\\data-5\\'#此处输入题目五经过matlab提取后的原图分割后的文件夹，请查看技术文档
    root2 = 'E:\\文档\\验证码识别\\处理后的数据\\data-5_match\\'#此处输入题目五经过matlab提取后的需要匹配图的文件夹，请查看技术文档
    model(IMAGE_HEIGHT=45, IMAGE_WIDTH=45, keep_prob_value=1.0, minibatch_size=256, pat=pat, root1=root1, root2=root2)

