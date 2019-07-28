# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:11:30 2018

@author: p0werHu
"""

import scipy.io as sio
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import scipy.io as sio  
import os
import math

def data_preprocess(root):
    train_data = np.zeros((10000, 60*200))
    for i in range(10000):
        img = mpimg.imread(root + '\\' + "%04d" % i + '.jpg')
        if (img.shape == (60,200)):
            train_data[i,:] = img.reshape(1,-1)
        else:
            print('图片大小有问题')
    sio.savemat(r'E:\文档\验证码识别\处理后的数据\data.mat', {'data': train_data}) 

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def creat_variable(C, IMAGE_HEIGHT, IMAGE_WIDTH):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name = 'X')
    Y = tf.placeholder(tf.float32, [None, C*5], name = 'Y')
    keep_prob = tf.placeholder(tf.float32) # dropout

    return X,Y,keep_prob

def convert_to_one_hot(Y, C):
    Y = Y.astype(int)
    temp_1 = np.eye(C)[Y[:,0].reshape(-1)].T
    temp_2 = np.eye(C)[Y[:,1].reshape(-1)].T
    temp_3 = np.eye(C)[Y[:,2].reshape(-1)].T
    temp_4 = np.eye(C)[Y[:,3].reshape(-1)].T
    temp_5 = np.eye(C)[Y[:,4].reshape(-1)].T
    temp = np.row_stack((temp_1, temp_2))
    temp = np.row_stack((temp, temp_3))
    temp = np.row_stack((temp, temp_4))
    Y = np.row_stack((temp,temp_5))
    return Y 

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

def forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob,C):
    
    with tf.name_scope('input_reshape'):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
 
	# 3 conv layer
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            w_c1 = tf.get_variable("w_c1",[3, 3, 1, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c1)
        with tf.name_scope('bias'):
            b_c1 = tf.get_variable("b_c1",[16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c1)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)
    
    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c2)
        with tf.name_scope('bias'):
            b_c2 = tf.get_variable("b_c2",[32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c2)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.name_scope('conv3'):
        with tf.name_scope('weights'):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c3)
        with tf.name_scope('bias'):
            b_c3 = tf.get_variable("b_c3",[64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c3)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    with tf.name_scope('conv4'):
        with tf.name_scope('weights'):
            w_c4 = tf.get_variable("w_c4",[3, 3, 64, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_c4)
        with tf.name_scope('bias'):
            b_c4 = tf.get_variable("b_c4",[128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_c4)
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob)
        tf.summary.histogram('conv4', conv4)
        
    r = conv4.get_shape()
	# Fully connected layer
    with tf.name_scope('FC'):
        with tf.name_scope('weights'):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_d)
        with tf.name_scope('bias'):
            b_d = tf.get_variable("b_d",[512],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_d)
        dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
    with tf.name_scope('output'):
        with tf.name_scope('weights'):
            w_out = tf.get_variable("w_out",[512,C*5], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(w_out)
        with tf.name_scope('bias'):
            b_out = tf.get_variable("b_out",[C*5], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            variable_summaries(b_out)
        out = tf.add(tf.matmul(dense, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    return out

def forward_propagation_lenet(w_alpha, b_alpha,IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob,C):
    
    with tf.name_scope('input_reshape'):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x, 100)
    #print(x.shape)
	#w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
	#w_c2_alpha = np.sqrt(2.0/(3*3*32)) 
	#w_c3_alpha = np.sqrt(2.0/(3*3*64)) 
	#w_d1_alpha = np.sqrt(2.0/(8*32*64))
	#out_alpha = np.sqrt(2.0/1024)
 
	# 3 conv layer
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            w_c1 = tf.Variable(w_alpha*tf.random_normal([5, 5, 1, 6]))
            variable_summaries(w_c1)
        with tf.name_scope('bias'):
            b_c1 = tf.Variable(b_alpha*tf.random_normal([6]))
            variable_summaries(b_c1)
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='VALID'), b_c1)
        conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.sigmoid(conv1)
        #conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)

    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            w_c2 = tf.Variable(w_alpha*tf.random_normal([5, 5, 6,16]))
            variable_summaries(w_c2)
        with tf.name_scope('bias'):
            b_c2 = tf.Variable(b_alpha*tf.random_normal([16]))
            variable_summaries(b_c2)
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='VALID'), b_c2)
        conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.sigmoid(conv2)
        #conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28

    #20*14
    r = conv2.get_shape()
	# Fully connected layer
    with tf.name_scope('FC'):
        with tf.name_scope('weights'):
            w_d = tf.Variable(w_alpha*tf.random_normal([r[1].value*r[2].value*r[3].value, 120]))
            variable_summaries(w_d)
        with tf.name_scope('bias'):
            b_d = tf.Variable(b_alpha*tf.random_normal([120]))
            variable_summaries(b_d)
        dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        #dense = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC', dense)
        
    with tf.name_scope('FC2'):
        with tf.name_scope('weights'):
            w_d2 = tf.Variable(w_alpha*tf.random_normal([120, 84]))
            variable_summaries(w_d2)
        with tf.name_scope('bias'):
            b_d2 = tf.Variable(b_alpha*tf.random_normal([84]))
            variable_summaries(b_d2)
        dense2 = tf.nn.sigmoid(tf.add(tf.matmul(dense, w_d2), b_d2))
        #dense2 = tf.nn.dropout(dense, keep_prob)
        tf.summary.histogram('FC2', dense2)
        
    with tf.name_scope('output'):
        with tf.name_scope('weights'):
            w_out = tf.Variable(w_alpha*tf.random_normal([84,C*5]))
            variable_summaries(w_out)
        with tf.name_scope('bias'):
            b_out = tf.Variable(b_alpha*tf.random_normal([C*5]))
            variable_summaries(b_out)
        out = tf.add(tf.matmul(dense2, w_out), b_out)
        tf.summary.histogram('output', out)
    #print(out.shape)
	#out = tf.nn.softmax(out)
    parameters = {"w_c1": w_c1,
                  "b_c1": b_c1,
                  "w_c2": w_c2,
                  "b_c2": b_c2,
                  "w_d2": w_d2,
                  "b_d2": b_d2,
                  "w_d": w_d,
                  "b_d": b_d,
                  "w_out": w_out,
                  "b_out": b_out,
                  }
    return out,parameters

def vec2text(temp):
    text_1 =  tf.reshape(tf.argmax(temp[:,0:36], 1),[-1,1])
    text_2 =  tf.reshape(tf.argmax(temp[:,36:72], 1),[-1,1])
    text_3 =  tf.reshape(tf.argmax(temp[:,72:108], 1),[-1,1])
    text_4 =  tf.reshape(tf.argmax(temp[:,108:144], 1),[-1,1])
    text_5 =  tf.reshape(tf.argmax(temp[:,144:180], 1),[-1,1])
    text =  tf.concat([text_1,text_2,text_3,text_4,text_5], 1)
    return text
    
def compare(predict, label):
    temp_1 = tf.equal(predict[:,0], label[:,0])
    temp_2 = tf.equal(predict[:,1], label[:,1])
    temp_3 = tf.equal(predict[:,2], label[:,2])
    temp_4 = tf.equal(predict[:,3], label[:,3])
    temp_5 = tf.equal(predict[:,4], label[:,4])
    correct_pred = tf.equal(temp_1,tf.equal(temp_2,tf.equal(temp_3, tf.equal(temp_4, temp_5))))
    return correct_pred

def model(w_alpha, b_alpha,IMAGE_HEIGHT, IMAGE_WIDTH, X_train, Y_train, C, 
          keep_prob_value, minibatch_size, X_val, Y_val):
    
    X,Y,keep_prob = creat_variable(C, IMAGE_HEIGHT, IMAGE_WIDTH)
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob, C)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        tf.summary.scalar('loss', loss)
    
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss)
    
    predict = vec2text(output)
    #print(predict.shape)
    label = vec2text(Y)
    #print(label.shape)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = compare(predict, label)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        
    merged = tf.summary.merge_all()
    
    seed = 2
    m = X_train.shape[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(r'E:\tensorboard\train', sess.graph)
        test_writer = tf.summary.FileWriter(r'E:\tensorboard\test')
        epoch = 0
        Y_val = convert_to_one_hot(Y_val, C).T
        Y_train = convert_to_one_hot(Y_train, C).T
        while True:
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    #print(minibatch_X.shape)
                    #print(minibatch_Y.shape)
                    _, minibatch_cost = sess.run([optimizer, loss], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: keep_prob_value})
                    epoch_cost += minibatch_cost / num_minibatches
            summary,_, minibatch_cost = sess.run([merged,optimizer, loss], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: keep_prob_value})
            train_writer.add_summary(summary, epoch)
            if epoch % 10 == 9:
                summary,acc_val = sess.run([merged,accuracy], feed_dict={X: X_val, Y: Y_val, keep_prob: 1.})
                test_writer.add_summary(summary, epoch)
                acc_train = sess.run(accuracy, feed_dict={X: X_train[0:200,:], Y: Y_train[0:200,:], keep_prob: 1.})
                print ("Val_Accuracy: %f" % (acc_val))
                print ("Train_Accuracy: %f" % (acc_train))
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            epoch += 1
            if epoch == 500:
                pat = r'E:\文档\验证码识别\model\task2'
                saver = tf.train.Saver()
                saver.save(sess, pat+"\\model.ckpt")
                print ("save parameters")
                break;
        train_writer.close()
        test_writer.close()
        print('end')
        
if __name__ == "__main__":
    #data_preprocess(r'E:\文档\验证码识别\处理后的数据\data-2')
    url = r'E:\文档\验证码识别\处理后的数据\subdata.mat'
    mat = sio.loadmat(url)
    data = mat['subdata']
    
    url = r'E:\文档\验证码识别\处理后的数据\sublabel.mat'
    mat = sio.loadmat(url)
    label = mat['sublabel']
    
    url = r'E:\文档\验证码识别\处理后的数据\valdata.mat'
    mat = sio.loadmat(url)
    X_val = mat['valdata']
    
    url = r'E:\文档\验证码识别\处理后的数据\vallabel.mat'
    mat = sio.loadmat(url)
    Y_label= mat['vallabel']
    
    parameters = model(0.01, 0.1,IMAGE_HEIGHT=60, IMAGE_WIDTH=200, X_train=data, Y_train=label, C=36, keep_prob_value=1.0, minibatch_size=64
          ,X_val=X_val,Y_val=Y_label) 