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
    filecount=0
    path = root + "%04d" % index
    ad = root
    for ad,dir,files in  os. walk(path):
        filecount+=len(files)

    train_data = np.zeros((filecount, 55*80))
    for j in range(filecount):
        img = mpimg.imread(path + '\\' + str(j+1) + '.jpg')
        if (img.shape[1] < 55):
            add_pixel = 55 - img.shape[1]
            img = np.lib.pad(img, ((0,0),(int(add_pixel/2),add_pixel-int(add_pixel/2))), 'constant', constant_values=(255,255))
            img = img.reshape((1,-1))
            train_data[j,:] = img
        else:
            print('大于55')
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
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='VALID'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
        tf.summary.histogram('conv1', conv1)

    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            w_c2 = tf.get_variable("w_c2",[3, 3, 16,32])
        with tf.name_scope('bias'):
            b_c2 = tf.get_variable("b_c2",[32])
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='VALID'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob)
        tf.summary.histogram('conv2', conv2)
    #40*28
    with tf.name_scope('conv3'):
        with tf.name_scope('weights'):
            w_c3 = tf.get_variable("w_c3",[3, 3, 32, 64])
        with tf.name_scope('bias'):
            b_c3 = tf.get_variable("b_c3",[64])
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='VALID'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, keep_prob)
        tf.summary.histogram('conv3', conv3)
    #20*14
    r = conv3.get_shape()
	# Fully connected layer
    with tf.name_scope('FC'):
        with tf.name_scope('weights'):
            w_d = tf.get_variable("w_d",[r[1].value*r[2].value*r[3].value, 512])
        with tf.name_scope('bias'):
            b_d = tf.get_variable("b_d",[512])
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
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

def transform(index):
    if index < 10:
        return str(index)
    elif index == 10:
        return '+'
    elif index == 11:
        return '-'
    elif index == 12:
        return '*'
    else :
        return 'error'
    
def model(IMAGE_HEIGHT, IMAGE_WIDTH, C, keep_prob_value, root, pat):
    
    X,keep_prob = creat_variable(C, IMAGE_HEIGHT, IMAGE_WIDTH)
    output = forward_propagation(IMAGE_HEIGHT, IMAGE_WIDTH, X, keep_prob, C)

    saver = tf.train.Saver()
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            predict =  tf.argmax(output, 1)
    
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pat+"\\model.ckpt")
        string = ""
        for i in range(5000): #此处输入样例数量，编号从0开始        
            test_data = data_preprocess(root,i)
            p = sess.run(predict, feed_dict={X: test_data, keep_prob: keep_prob_value})
            f = ''
            for j in range(p.shape[0]):
                f += transform(p[j])
            res=cal(f)
            string += str("%04d" % i) +','+ f + '=' + str(int(res)) + '\n'
        fout = open('mappings.txt','wt')
        fout.write(string)
        fout.close()
        print('end')

def isNum(value):
    try:
        value+1
    except TypeError:
        return False
    else:
        return True

def cal(strexpp):
    seqLegalChar=('+','-','*','/','(',')','^','1','2','3','4','5','6','7','8','9','0','.')  

    seqOpr=seqLegalChar[0:7]                   
    seqOpa=seqLegalChar[7:18]                   
    
    if len(strexpp)==0: return Null
    
    strexpr=strexpp.replace(' ','')
   
    for ch in strexpr:
        if ch not in seqLegalChar: 
            return Null
  
    numstart=-1                                 
    seqExpr=[]                                 
    for i in range(0,len(strexpr)):            
       
        if (strexpr[i] in seqOpa):                             
            if numstart<0:                      
                numstart=i
            if i==len(strexpr)-1:           
                seqExpr.append(float(strexpr[numstart:len(strexpp)]))
            continue                                
        
       
        if numstart>=0:                
            seqExpr.append(float(strexpr[numstart:i]))  
        seqExpr.append(strexpr[i])         
        numstart=-1                          
    
    
    seqPosfix=[]                                
    stkOpra=[]                                  
    for op in seqExpr:
        if isNum(op):               
            seqPosfix.append(op)
            continue
        if not stkOpra:                   
            stkOpra.append(op)
            continue
        if op ==')':                            
            while stkOpra[len(stkOpra)-1]!='(':
                seqPosfix.append(stkOpra.pop())      
            stkOpra.pop()
        elif op in ['+','-']:               
            while (stkOpra and (stkOpra[len(stkOpra)-1]!='(')):
                seqPosfix.append(stkOpra.pop())
            stkOpra.append(op)
        elif op in ['*','/']:              
            while (stkOpra and (stkOpra[len(stkOpra)-1]=='^' )):
                seqPosfix.append(stkOpra.pop())
            stkOpra.append(op)
        elif op in ['^','(']:               
            stkOpra.append(op)
    while stkOpra:                         
        seqPosfix.append(stkOpra.pop())
   
    stkNumb=[]                             
    for op in seqPosfix:
        if isNum(op):
            stkNumb.append(op)     
            continue
        p1=stkNumb.pop()           
        p2=stkNumb.pop()        
        if op=='^':                 
            p=p2**p1
        elif op=='*':
            p=p2*p1
        elif op=='/':
            p=p2/p1
        elif op=='+':
            p=p2+p1
        elif op=='-':
            p=p2-p1
        stkNumb.append(p)          
    
    return stkNumb.pop() 

if __name__ == "__main__":
    global Null  
    Null=''
    pat = 'E:\\文档\\验证码识别\\model\\task1' #此处输入题目一的模型地址
    root = 'E:\\文档\\验证码识别\\处理后的数据\data-1\\'#此处输入经过matlab分割之后问图片文件夹，请参阅部署文档
    model(IMAGE_HEIGHT=80, IMAGE_WIDTH=55, C=13, keep_prob_value=0.75, root=root, pat=pat)
