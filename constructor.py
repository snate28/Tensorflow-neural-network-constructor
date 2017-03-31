# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:23:07 2017

@author: snate
"""
from scipy.misc import imread, imresize
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pickle
import PIL
#Defining hyperparameters
img_width = 100
img_height = 100
num_channels=3

filter_size1=5
num_filters1=20

filter_size2=5
num_filters2=40

filter_size3=5
num_filters3=40

filter_size4=5
num_filters4=40

num_neurons_fully_connected_layer=150


def init_weights(shape,stddev=0.01):
    
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    
def biases_init(size):
    
    return tf.Variable(tf.constant(0.05,shape=[size]))


def conv_layer(input,num_channels,filter_size,num_filters,pooling=True):
    
    weights = init_weights([filter_size,filter_size,num_channels,num_filters])
    
    biases = biases_init(num_filters)
    
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
    
    layer = layer+biases
    
    if pooling:
        
        layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    layer = tf.nn.relu(layer)
    
    return layer, weights
    

def flatten(layer):
    features = layer.get_shape()[1:4].num_elements()
    
    flattened = tf.reshape(layer,[-1,features])
    
    return flattened, features
    

def fully_connected(input,num_in,num_out,relu=True):
    weights = init_weights(shape=[num_in,num_out])
    
    biases = biases_init(num_out)
    
    layer = tf.matmul(input,weights)+biases
    
    if relu:
        layer = tf.nn.relu(layer)
    
    return layer, weights
 



def inflate(image,h=50,w=50,crop=True):
    if crop:
        image = tf.random_crop(image, size=[h, w, num_channels])
    
    image = tf.image.random_flip_left_right(image)
    
    tf.image.random_flip_up_down(image, seed=None)    
    
    image = tf.image.random_hue(image, max_delta=0.05)
    
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    #image = tf.minimum(image, 1.0)
    #image = tf.maximum(image, 0.0)
    #image=resize_image(image,(h,w))
    image = tf.image.resize_images(image,img_height,img_width)



    
    return image





train_x = pickle.load(open('/media/snate/58C6F68EC6F66C1E/machine learning/traintrue.txt','rb'))
train_y = pickle.load(open('/media/snate/58C6F68EC6F66C1E/machine learning/labelstrue.txt','rb'))
dataset_size=len(train_x)
#this is a kostyl
train_x.append([0])
train_y.append([0])

iterations_made = 0

def train(dataset,iterations,batch_size):
    global iterations_made
    index = 0
    for i in range(iterations_made, iterations+iterations_made):
        
        if dataset_size-index-batch_size > 0:
            x_batch = np.asanyarray(dataset[index:index+batch_size])
            print x_batch.shape
            x_batch = np.reshape(x_batch, (-1, img_width, img_width, num_channels))
            y_batch = train_y[index:index+batch_size]
           
            index = index + batch_size
            
        else:
            
            x_batch = dataset[index:index+1]
            x_batch = np.reshape(x_batch, (-1, img_width, img_width, num_channels))
            y_batch = train_y[index:index+1]
            
            index=index+1
            
            if dataset_size-index==0:
                index = 0
            
        
        feed = {x: x_batch, y_true: y_batch}
        
        session.run(optimizer, feed_dict=feed)
        iterations_made=iterations_made+1
        
        if i % 5000 == 0:
            accur = session.run(accuracy, feed_dict=feed)
            print"Current iteration: {0:>6}, accuracy: {1:>6.1%}".format(i + 1, accur)
             
        
        print i



        




#Visualisations

def plot_conv_weights(weights, input_channel=1):
   
    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = int(math.ceil(math.sqrt(num_filters)))
    print num_grids
    print type(num_grids)
    # grid of plots
    fig, axes = plt.subplots(num_grids, num_grids)

    
    for i, ax in enumerate(axes.flat):
       
        if i<num_filters:
            
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    




def plot_conv_layer(layer, image):

    feed_dict = {x: [image]}

    
    values = session.run(layer, feed_dict=feed_dict)

    num_filters = values.shape[3]

    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    
    fig, axes = plt.subplots(num_grids, num_grids)

    
    for i, ax in enumerate(axes.flat):
        
        if i<num_filters:
            
            img = values[0, :, :, i]

            
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    



def test(iterations,batch_size):
    global iterations_made
    index = 0
    accur=[]
    for i in range(iterations_made, iterations+iterations_made):
        
        if dataset_size-index-batch_size > 0:
            x_batch = train_x[index:index+batch_size]
            x_batch = np.reshape(x_batch, (-1, img_width, img_width, num_channels))
            y_batch = train_y[index:index+batch_size]
            
            index = index + batch_size
            
        else:
            
            x_batch = train_x[index:index+1]
            x_batch = np.reshape(x_batch, (-1, img_width, img_width, num_channels))
            y_batch = train_y[index:index+1]
            
            index=index+1
            
            if dataset_size-index==0:
                index = 0
            
        
        feed = {x: x_batch, y_true: y_batch}
        
        accur.append(session.run(accuracy, feed_dict=feed))
        
        print i
    accurac=sum(accur)/len(accur)
    print accurac
    
def raw_pred(img,w=100,h=100):
    img = imread(img)
    img = np.asanyarray(imresize(img,(w,h)))
    #a1=img[:,:,2].reshape(100,100,1)
    #a2=img[:,:,1].reshape(100,100,1)
    #a3=img[:,:,0].reshape(100,100,1)
    #img = np.concatenate((a3,a2,a1),axis=2)
    img=np.reshape(img,(-1,w,h,3))
    #del a1,a2,a3
    imshow(img.reshape(100,100,3))
    print session.run(pred,feed_dict={x:img})


def raw_plot(img,target_layer,w=100,h=100):
    img = imread(img)
    img = np.asanyarray(imresize(img,(w,h)))
    img=np.reshape(img,(w,h,3))
    print img.shape
    plot_conv_layer(target_layer,img)
    
    
def imshow(img):
    plt.imshow(img)
    #im = im.astype(np.uint8) for the case of showing the resized image convertion to int is required


#Constructing the neural network

#Defining placeholders
x = tf.placeholder(tf.float32, shape=[None, img_width, img_width, num_channels], name='x')




y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)




conv1, weights_conv1 = conv_layer(input=x, num_channels=3, filter_size=filter_size1, num_filters=num_filters1, pooling=True)
conv2, weights_conv2 = conv_layer(input=conv1, num_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, pooling=True)    
conv3, weights_conv3 = conv_layer(input=conv2, num_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, pooling=True)   
conv4, weights_conv4 = conv_layer(input=conv3, num_channels=num_filters3, filter_size=filter_size4, num_filters=num_filters4, pooling=True)       

flattened_layer, features = flatten(conv4)

fully_connected_layer, fully_connected_weights = fully_connected(flattened_layer, features, num_neurons_fully_connected_layer)

last, last_weights = fully_connected(fully_connected_layer,num_neurons_fully_connected_layer,2, relu = False)

pred=tf.nn.softmax(last)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#This is accuracy
pred_class=tf.argmax(pred,dimension=1)
true_class=tf.argmax(y_true,dimension=1)
correct = tf.equal(pred_class,true_class)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())
    

#inflated_set = tf.map_fn(lambda image: inflate(image), train_x)
    
#print test(100,10)
#print session.run(last,feed_dict={x:train_x[0:1]})
saver = tf.train.Saver()
for i in range(50):
    #train(inflated_set,100,50)
    train(train_x,100,50)
    saver.save(sess=session,save_path="model.ckpt")
    print "THIS IS" ,i ,"TIME THOUGHT THE DATASET_____________"
#plot_conv_weights(weights=weights_conv3)
#plot_conv_layer(conv1, image=train_x[0])
#saver.restore(sess=session, save_path='model.ckpt')

#session.run(pred,feed_dict={x:train_x[0:1]})


