# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:59:34 2018

@author: p0werHu
"""

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import time
import random
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
    
    writer = tf.python_io.TFRecordWriter('task5_test.tfrecords')
    class_path = 'E:\\文档\\验证码识别\\处理后的数据\\data-5_triple\\'
    for i in range(300000,320000,1):
        if i%10000 == 0:
            print(i)
        triple_data_path = class_path + str(i) + '\\'
        anchor =  picture2string(1, triple_data_path)
        pos = picture2string(2, triple_data_path)
        neg = picture2string(3, triple_data_path)
        example = tf.train.Example(features=tf.train.Features(feature={
                'anchor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[anchor])),
                'positive': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos])),
                'negative': tf.train.Feature(bytes_list=tf.train.BytesList(value=[neg]))
                }))
        writer.write(example.SerializeToString())
    writer.close()
            
def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'anchor' : tf.FixedLenFeature([], tf.string),
                                               'positive' : tf.FixedLenFeature([], tf.string),
                                               'negative' :  tf.FixedLenFeature([], tf.string)
                                               })
    anc = tf.decode_raw(features['anchor'], tf.uint8)
    anc = tf.reshape(anc, [45,45])
    pos = tf.decode_raw(features['positive'], tf.uint8)
    pos = tf.reshape(pos, [45,45])
    neg = tf.decode_raw(features['negative'], tf.uint8)
    neg = tf.reshape(neg, [45,45])
    min_after_dequeue = 400
    num_threads = 3
    capacity = min_after_dequeue + num_threads * batch_size
    anc_batch, pos_batch, neg_batch = tf.train.shuffle_batch([anc, pos, neg], batch_size=batch_size,capacity=
                                                             capacity, min_after_dequeue=min_after_dequeue, num_threads=
                                                             num_threads)
    
    return anc_batch,pos_batch,neg_batch
#--------------------------------------------------------------------------------------------------------
    

#------------------------------------training---------------------------------------------------------

def creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH):
    anchor = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name = 'anchor')
    positive = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name = 'positive')
    negative = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name = 'negative')
    keep_prob = tf.placeholder(tf.float32) # dropout
    learning_rate = tf.placeholder(tf.float32)

    return anchor, positive, negative, keep_prob, learning_rate

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
            w_c1 = tf.get_variable("w_c1",[3, 3, 1, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c1)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c1 = tf.get_variable("b_c1",[16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c1)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)
    
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c2)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c2 = tf.get_variable("b_c2",[32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c2)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c3)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c3 = tf.get_variable("b_c3",[64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c3)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    
    with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_c4 = tf.get_variable("w_c4",[3, 3, 64, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c4)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_c4 = tf.get_variable("b_c4",[128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c4)
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob)
        tf.summary.histogram('conv4', conv4)
        
    r = conv4.get_shape()
	# Fully connected layer
    with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_d)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_d = tf.get_variable("b_d",[512],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_d)
        dense = tf.reshape(conv4,[-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
    
    with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights',reuse=tf.AUTO_REUSE):
            w_out = tf.get_variable("w_out",[512,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_out)
        with tf.variable_scope('bias',reuse=tf.AUTO_REUSE):
            b_out = tf.get_variable("b_out",[64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_out)
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    parameters = {"w_c1": w_c1,
                  "b_c1": b_c1,
                  "w_c2": w_c2,
                  "b_c2": b_c2,
                  "w_c3": w_c3,
                  "b_c3": b_c3,
                  "w_d": w_d,
                  "b_d": b_d,
                  "w_out": w_out,
                  "b_out": b_out,
                  }
    return out, parameters

def forward_propagation_lenet(w_alpha, b_alpha,IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob):
    
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
    parameters = None
    return out, parameters

def compute_loss(anchor, positive, negative, margin):
    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

def get_training_data(sess, anc, pos, neg):
    anc_batch, pos_batch, eng_batch = sess.run([anc, pos, neg])
    return anc_batch, pos_batch, eng_batch

def model(IMAGE_HEIGHT, IMAGE_WIDTH):
    
    
    tf.reset_default_graph()
    anchor, positive, negative, keep_prob, learning_rate = creat_variable(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    anchor_output, _ = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, anchor, keep_prob)
    #anchor_output = anchor_output / tf.nn.l2_normalize(anchor_output,axis=1,keepdims=True)
    positive_output, _ = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, positive, keep_prob)
    #positive_output = positive_output / tf.nn.l2_normalize(positive_output,axis=1,keepdims=True)
    negative_output, parameters = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, negative, keep_prob)
    #negative_output = negative_output / tf.norm(negative_output,axis=1,keepdims=True)
    with tf.variable_scope('loss'):
        loss = compute_loss(anchor_output, positive_output, negative_output, margin=1)
        tf.summary.scalar('loss', loss)
    
#    with tf.variable_scope('distant'):
#        with tf.variable_scope('positive'):
#            dis_pos = tf.reduce_sum(tf.square(anchor_output-positive_output),1)
#            variable_summaries(dis_pos)
#        with tf.variable_scope('negative'):
#            dis_neg = tf.reduce_sum(tf.square(negative_output-anchor_output),1)
#            variable_summaries(dis_neg)
    
    
#    accuracy_pos = tf.reduce_mean(tf.cast(dis_pos<16, tf.float32))
#    accuracy_neg = tf.reduce_mean(tf.cast(dis_neg>16 , tf.float32))
#    
#    with tf.variable_scope('accuracy'):
#        accuracy = (accuracy_pos + accuracy_neg) / 2
#        tf.summary.scalar('accuracy', accuracy)
    
    #merged = tf.summary.merge_all()
    
    with tf.variable_scope('train'):
        #global_step = tf.Variable(0, trainable=False)
        #initial_learning_rate = 0.0001
        #earning_rate = tf.train.exponential_decay(initial_learning_rate,
        #                                          global_step=global_step,
        #                                          decay_steps=7500,decay_rate=0.95)
        #tf.summary.scalar('learning_rate', earning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    return optimizer, loss, anchor, positive, negative, keep_prob, learning_rate
    
    
def train(optimizer, loss, max_epoch, minibatch_size, learning_rate_v, anchor, positive, negative, keep_prob, learning_rate):
    anc,pos,neg = read_and_decode(r'task5_train.tfrecords', minibatch_size)
    anc_val,pos_val,neg_val = read_and_decode(r'task5_test.tfrecords', 2000)
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model/model.ckpt")
        #train_writer = tf.summary.FileWriter(r'E:\tensorboard\5\train', sess.graph)
        #test_writer = tf.summary.FileWriter(r'E:\tensorboard\5\test')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord) 
        iterator = 0 
        #iterator_cost = 0.
        while True:
            iterator += 1 
            iterator_cost = 0
            epoch_num = int(300000 / minibatch_size)
            for i in range(epoch_num):
                anc_batch, pos_batch, neg_batch = get_training_data(sess, anc, pos, neg)
                _, minibatch_cost = sess.run([optimizer, loss], feed_dict={anchor: anc_batch
                                            , positive: pos_batch, negative: neg_batch, keep_prob: 0.8, learning_rate: learning_rate_v})
            #anc_batch, pos_batch, neg_batch = get_training_data(sess, anc, pos, neg)
            #po,ne = sess.run([dis_pos, dis_neg], feed_dict={anchor: anc_batch
             #                                , positive: pos_batch, negative: neg_batch, keep_prob: keep_prob_value})
           ## summary,_, _ = sess.run([merged,optimizer, loss], feed_dict={anchor: anc_batch
           ##                                  , positive: pos_batch, negative: neg_batch, keep_prob: keep_prob_value})
            #train_writer.add_summary(summary, iterator)
                #if i%50 == 49:
            if iterator == max_epoch:#iterator >= max_epoch-20:
                epoch_num = int(20000 / minibatch_size)
                for i in range(epoch_num):
                    anc_val_batch, pos_val_batch, neg_val_batch = get_training_data(sess, anc_val, pos_val, neg_val)
                    minibatch_cost = sess.run(loss,feed_dict={anchor: anc_val_batch, positive: pos_val_batch, negative: neg_val_batch, keep_prob: 1., learning_rate: learning_rate_v})
                    iterator_cost += minibatch_cost/epoch_num
                break
                #iterator_cost  = iterator_cost * 0.9 + iterator_cost_t * 0.1
                #print(str(iterator) +'  '+ str(iterator_cost))
            #test_writer.add_summary(summary, iterator)
            #print("Cost after iterator %i: %f" % (iterator, iterator_cost))
            #print("Accuracy : %f" % (accuracy_train))
            #if iterator == max_epoch:
            #    break
            #if iterator % 30 == 29:
            #    pat = 'E:\\文档\\验证码识别\\task5_modle\\'+str(iterator) + 'accuracy'+str(iterator_cost)
            #    os.makedirs(pat)
            #    saver = tf.train.Saver()
            #    saver.save(sess, pat+"\\model.ckpt")
            #    print(str(iterator)+'save')
#            if iterator == 180:
#                pat = 'E:\\文档\\验证码识别\\task5_modle\\'+'最终accuracy'+str(accuracy_train)
#                os.makedirs(pat)
#                saver = tf.train.Saver()
#                saver.save(sess, pat+"\\model.ckpt")
#                print('end save')
#                break
        coord.request_stop()
        coord.join(threads)
        
        #train_writer.close()
        #test_writer.close()
        #print('end')
    return iterator_cost
#-----------------------------------------end-----------------------------------------------------
#-------------------------------------Cross-validation---------------------------------------------------
def get_val_accuracy():
    pass

def function(x):
    return (x*x + 2*x)/100000

if __name__ == '__main__':
    #create_tfrecord()
#   coefficient
    fi = open('Results.txt', 'w')
    c1_start = 1.494
    c1_end = 1.494
    c2_start = 1.494
    c2_end = 1.494
    Vmax = 0.001
    itermax = 50
    w_start = 0.9
    w_end = 0.4
    
    x = [0.00022,0.00018,0.00019,0.0002,0.00008]
    v = [0.0001,0.0008,0.0009,0.0002,0.00018]
    cost = [100,100,100,100,100]
    gbest_value = 10000
    gbest = 0
    pbest_value = [10000,10000,10000,10000,10000]
    pbest = [0,0,0,0,0]
    optimizer, loss, anchor, positive, negative, keep_prob, learning_rate = model(IMAGE_HEIGHT=45, IMAGE_WIDTH=45)
    #_ = train(optimizer, loss, 116, 128, 0.0001, anchor, positive, negative, keep_prob, learning_rate)
    for iteration in range(1,25):
        fi.write('第%d次迭代\n' %(iteration))
        print('第%d次迭代' %(iteration))
        w = w_start - (w_start - w_end) * iteration/itermax
        c1 = c1_start - (c1_start - c1_end) * iteration/itermax
        c2 = c2_start + (c2_end - c2_start) * iteration/itermax
        for i in range(5):
            cost[i] = train(optimizer, loss, 25, 128, x[i], anchor, positive, negative, keep_prob, learning_rate)
            #cost[i] = function(x[i])
            if cost[i]<pbest_value[i]:
                pbest[i] = x[i] 
                pbest_value[i] = cost[i] 
        for i in range(5):
            if pbest_value[i] < gbest_value:
                gbest_value = pbest_value[i]
                gbest = pbest[i]
        for i in range(5):
            v[i] = w*v[i] + c1*random.random()*(pbest[i]-x[i]) + c2*random.random()*(gbest-x[i])
            if v[i]>Vmax:
                v[i] = Vmax
            x[i] = x[i] + v[i]
            if x[i] < 0.:
                x[i] = -1 * x[i]
            print('粒子：%d, 位置：%f, 损失函数:%f'% (i,x[i],cost[i]))
            fi.write('粒子：%d, 位置：%f, 损失函数:%f\n'% (i,x[i],cost[i]))
    fi.close()
    #anc,pos,neg = read_and_decode(r'E:\文档\验证码识别\task5_train.tfrecords',1000)
    #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        #anc_batch, pos_batch, eng_batch = sess.run([anc, pos, neg])
        #coord.request_stop()
        #coord.join(threads)
