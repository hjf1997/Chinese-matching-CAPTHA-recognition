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
    img_raw = img.tostring()
    return img_raw

def create_tfrecord():
    
    writer = tf.python_io.TFRecordWriter('task5_finally.tfrecords')
    class_path = 'E:\\文档\\验证码识别\\处理后的数据\\data-5\\'
    match_path = 'E:\\文档\\验证码识别\\处理后的数据\\data-5_match\\'
    for i in range(9500,10000,1):
        if i%50 == 0:
            print(i)
        data_path = class_path + "%04d" % i + '\\'
        data_match_path = match_path + "%04d" % i + '\\'
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
        example = tf.train.Example(features=tf.train.Features(feature={
                'data1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data1])),
                'data2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data2])),
                'data3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data3])),
                'data4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data4])),
                'match0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match0])),
                'match1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match1])),
                'match2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match2])),
                'match3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match3])),
                'match4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match4])),
                'match5': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match5])),
                'match6': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match6])),
                'match7': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match7])),
                'match8': tf.train.Feature(bytes_list=tf.train.BytesList(value=[match8]))
                }))
        writer.write(example.SerializeToString())
    writer.close()
            
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'data1' : tf.FixedLenFeature([], tf.string),
                                               'data2' : tf.FixedLenFeature([], tf.string),
                                               'data3' :  tf.FixedLenFeature([], tf.string),
                                               'data4' :  tf.FixedLenFeature([], tf.string),
                                               'match0' :  tf.FixedLenFeature([], tf.string),
                                               'match1' :  tf.FixedLenFeature([], tf.string),
                                               'match2' :  tf.FixedLenFeature([], tf.string),
                                               'match3' :  tf.FixedLenFeature([], tf.string),
                                               'match4' :  tf.FixedLenFeature([], tf.string),
                                               'match5' :  tf.FixedLenFeature([], tf.string),
                                               'match6' :  tf.FixedLenFeature([], tf.string),
                                               'match7' :  tf.FixedLenFeature([], tf.string),
                                               'match8' :  tf.FixedLenFeature([], tf.string),
                                               })
    data1 = tf.decode_raw(features['data1'], tf.uint8)
    data1 = tf.reshape(data1, [45,45])
    data2 = tf.decode_raw(features['data2'], tf.uint8)
    data2 = tf.reshape(data2, [45,45])
    data3 = tf.decode_raw(features['data3'], tf.uint8)
    data3 = tf.reshape(data3, [45,45])
    data4 = tf.decode_raw(features['data4'], tf.uint8)
    data4 = tf.reshape(data4, [45,45])
    match0 = tf.decode_raw(features['match0'], tf.uint8)
    match0 = tf.reshape(match0, [45,45])
    match1 = tf.decode_raw(features['match1'], tf.uint8)
    match1 = tf.reshape(match1, [45,45])
    match2 = tf.decode_raw(features['match2'], tf.uint8)
    match2 = tf.reshape(match2, [45,45])
    match3 = tf.decode_raw(features['match3'], tf.uint8)
    match3 = tf.reshape(match3, [45,45])
    match4 = tf.decode_raw(features['match4'], tf.uint8)
    match4 = tf.reshape(match4, [45,45])
    match5 = tf.decode_raw(features['match5'], tf.uint8)
    match5 = tf.reshape(match5, [45,45])
    match6 = tf.decode_raw(features['match6'], tf.uint8)
    match6 = tf.reshape(match6, [45,45])
    match7 = tf.decode_raw(features['match7'], tf.uint8)
    match7 = tf.reshape(match7, [45,45])
    match8 = tf.decode_raw(features['match8'], tf.uint8)
    match8 = tf.reshape(match8, [45,45])
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
#    min_after_dequeue = 400
#    num_threads = 3
#    capacity = min_after_dequeue + num_threads * batch_size
#    anc_batch, pos_batch, neg_batch = tf.train.shuffle_batch([anc, pos, neg], batch_size=batch_size,capacity=
#                                                             capacity, min_after_dequeue=min_after_dequeue, num_threads=
#                                                             num_threads)
    
    return data
#--------------------------------------------------------------------------------------------------------
    

#------------------------------------training---------------------------------------------------------

def creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH):
    data = tf.placeholder(tf.float32, [IMAGE_HEIGHT, IMAGE_WIDTH], name = 'data')
    keep_prob = tf.placeholder(tf.float32) # dropout
    
    return data, keep_prob

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.variable_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.variable_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

def forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob):
    
    with tf.variable_scope('input_reshape',reuse=tf.AUTO_REUSE):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
	# 3 conv layer
    x = x / 255. - 0.3
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c1 = tf.get_variable("w_c1",[3, 3, 1, 16])
            variable_summaries(w_c1)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c1 = tf.get_variable("b_c1",[16])
            variable_summaries(b_c1)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)
    
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32])
            variable_summaries(w_c2)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c2 = tf.get_variable("b_c2",[32])
            variable_summaries(b_c2)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64])
            variable_summaries(w_c3)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c3 = tf.get_variable("b_c3",[64])
            variable_summaries(b_c3)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    
    with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c4 = tf.get_variable("w_c4",[3, 3, 64, 128])
            variable_summaries(w_c4)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c4 = tf.get_variable("b_c4",[128])
            variable_summaries(b_c4)
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob)
        tf.summary.histogram('conv4', conv4)
        
#    with tf.variable_scope('conv5',reuse=tf.AUTO_REUSE):
#        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
#            w_c5 = tf.get_variable("w_c5",[3, 3, 64, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#            variable_summaries(w_c5)
#        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
#            b_c5 = tf.get_variable("b_c5",[128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#            variable_summaries(b_c5)
#        conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
#        conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#        conv5 = tf.nn.dropout(conv5, keep_prob)
#        tf.summary.histogram('conv5', conv5)
#        
#    with tf.variable_scope('conv6',reuse=tf.AUTO_REUSE):
#        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
#            w_c6 = tf.get_variable("w_c6",[3, 3, 128, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#            variable_summaries(w_c5)
#        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
#            b_c6 = tf.get_variable("b_c6",[256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#            variable_summaries(b_c6)
#        conv6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME'), b_c6))
#        conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#        conv6 = tf.nn.dropout(conv6, keep_prob)
#        tf.summary.histogram('conv6', conv6)
        
    r = conv4.get_shape()
	# Fully connected layer
    with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512])
            variable_summaries(w_d)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_d = tf.get_variable("b_d",[512])
            variable_summaries(b_d)
        dense = tf.reshape(conv4,[-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
    
    with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_out = tf.get_variable("w_out",[512,64])
            variable_summaries(w_out)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_out = tf.get_variable("b_out",[64])
            variable_summaries(b_out)
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    return out

def forward_propagation_lenet(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob):
    
    with tf.variable_scope('input_reshape',reuse=tf.AUTO_REUSE):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
	# 3 conv layer
    x = x / 255. - 0.3
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c1 = tf.get_variable("w_c1",[5, 5, 1, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c1)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c1 = tf.get_variable("b_c1",[6], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c1)
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
        conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.sigmoid(conv1)
        tf.summary.histogram('conv1', conv1)
    
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c2 = tf.get_variable("w_c2",[5, 5, 6,16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c2)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c2 = tf.get_variable("b_c2",[16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c2)
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
        conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.sigmoid(conv2)
        tf.summary.histogram('conv2', conv2)
    #40*28
    

        
    r = conv2.get_shape()
	# Fully connected layer
    with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 120], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_d)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_d = tf.get_variable("b_d",[120],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_d)
        dense = tf.reshape(conv2,[-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.sigmoid(tf.add(tf.matmul(dense, w_d), b_d))
        tf.summary.histogram('FC', dense)
        
    with tf.variable_scope('FC2',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_d2 = tf.get_variable("w_d2",[120, 84], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_d2)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_d2 = tf.get_variable("b_d2",[84],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_d2)
        dense2 = tf.nn.sigmoid(tf.add(tf.matmul(dense, w_d2), b_d2))
        tf.summary.histogram('FC', dense)
    
    with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_out = tf.get_variable("w_out",[84,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_out)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_out = tf.get_variable("b_out",[64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_out)
        out = tf.add(tf.matmul(dense2, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    return out

def get_training_data(sess, data):
    data_o = sess.run(data)
    return data_o

def model(IMAGE_HEIGHT, IMAGE_WIDTH, keep_prob_value, minibatch_size):
    
    data,keep_prob = creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, data, keep_prob)
    
    test_data = read_and_decode(r'E:\文档\验证码识别\task5_finally.tfrecords')
    saver = tf.train.Saver()  
    
    url = r'E:\文档\验证码识别\参数及数据集\data-5\label.mat'
    mat = sio.loadmat(url)
    label = mat['label']
    print('xixi')
    
    with tf.Session() as sess:
        pat = 'E:\\文档\\验证码识别\\model\\task5'
        saver.restore(sess, pat+"\\model.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        predicts = []
        for it in range(500):
            data_test = get_training_data(sess, test_data)
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
                #predict[row] = col
                #print('图'+str(row)+'匹配'+str(col))
            #print(predict)
            predicts.append(predict)
        p = np.array(predicts)
        label = label.reshape(-1)
        accuracy = np.sum(p == label[9500:10000]) / 500
        print('accuracy:'+str(accuracy))
        coord.request_stop()
        coord.join(threads)
#-----------------------------------------end-----------------------------------------------------
#-------------------------------------Cross-validation---------------------------------------------------


if __name__ == '__main__':
    #reate_tfrecord()
    model(IMAGE_HEIGHT=45, IMAGE_WIDTH=45, keep_prob_value=0.8, minibatch_size=256)
    #anc,pos,neg = read_and_decode(r'E:\文档\验证码识别\task5_train.tfrecords',1000)
    #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        #anc_batch, pos_batch, eng_batch = sess.run([anc, pos, neg])
        #coord.request_stop()
        #coord.join(threads)
