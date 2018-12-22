import os
import random
import sys
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
from scipy import ndimage
from PIL import Image
import os 
import numpy as np
from pandas import DataFrame
import xlsxwriter
import json
#from load_dataset1 import *



def initialize(layer_dims):
    parameters={}
    np.random.seed(1)
    for i in range(1,len(layer_dims)):
        parameters['W'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*(2/np.sqrt(layer_dims[i-1]))
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
    

    return parameters

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A,Z

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    
    return A, Z
    
        

def linear_forward(wpass,bpass,A):
    Z=np.add(np.matmul(wpass,A),bpass)
    cache=(A,wpass,bpass)
    return Z,cache

def l_model_forward(train_x,parameters,layer_dims):
    A=train_x
    caches=[]
    for i1 in range(1,len(layer_dims)-1):
        A_prev=A
        
        wpass=parameters['W'+str(i1)]
        
        bpass=parameters['b'+str(i1)]
        Z,linear_cache=linear_forward(wpass,bpass,A_prev)
        
        
        A,activation_cache=sigmoid(Z)
        cache=(linear_cache,activation_cache)
        caches.append(cache)
    
    i1=i1+1
    wpass=parameters['W'+str(i1)]
    bpass=parameters['b'+str(i1)]
    Z,linear_cache=linear_forward(wpass,bpass,A)
    A,activation_cache=sigmoid(Z)
    cache=(linear_cache,activation_cache)
    caches.append(cache)
    
    return A,caches

def compute_cost(Af,train_y):
    k1=train_y.shape[1]
    cost=(-1/k1)*(np.dot(train_y,np.log(Af).T)+np.dot((1-train_y),np.log(1-Af).T))
    return cost

def relu_backward(dAl,activation_cache,train_y):
    Z=activation_cache
    dz=np.array(dAl,copy=True)
    dz[Z <=0] = 0
    return dz


def sigmoid_backward(dal,activation_cache,train_y):
    Z=activation_cache
    s=1/(1+np.exp(-Z))
    return dal*s*(1-s)

def linear_backward2(dz,linear_cache,layer_dims,train_y):
    l=train_y.shape[1]
    A_prev,w_inter,b_inter=linear_cache
    dw=(1.0/l)*np.dot(dz,A_prev.T)
    db=(1.0/l)*np.sum(dz,axis=-1,keepdims=True)
    da_prev=np.dot(w_inter.T,dz)
    return dw,db,da_prev



def linear_backward(linear_cache,activation_cache,dAl,activationuse,layer_dims,train_y):
    if activationuse=="relu":
        dz=relu_backward(dAl,activation_cache,train_y)
        dww,dbb,daa=linear_backward2(dz,linear_cache,layer_dims,train_y)
    else:
        dz=sigmoid_backward(dAl,activation_cache,train_y)
        dww,dbb,daa=linear_backward2(dz,linear_cache,layer_dims,train_y)
    return dww,dbb,daa


def l_model_backprop(caches,Af,train_y,layer_dims):
    r=-(train_y/Af-(1-train_y)/(1-Af))
    L=len(layer_dims)
    L=L-1
    dAL=r
    grads={}
    d=len(layer_dims)
    cachecurr=caches[d-2]
    linear_cache,activation_cache=cachecurr

    grads['dW'+str(len(layer_dims)-1)],grads['db'+str(len(layer_dims)-1)],grads['dA'+str(len(layer_dims)-1)]=linear_backward(linear_cache,activation_cache,r,"sigmoid",layer_dims,train_y)
    for p in reversed(range(0,len(layer_dims)-2)):
        cachecurr=caches[p]
        linear_cache,activation_cache=cachecurr
        grads['dW'+str(p+1)],grads['db'+str(p+1)],grads['dA'+str(p+1)]=linear_backward(linear_cache,activation_cache,grads['dA'+str(p+2)],"sigmoid",layer_dims,train_y)
    return grads


def optimize(grads,parameters,layer_dims,learning_rate):
    for i in range(1,len(layer_dims)):
        parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*grads['dW'+str(i)]
        parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*grads['db'+str(i)]
    return parameters

def l_layer_nn(train_x,train_y,learning_rate,layer_dims,iter):
    np.random.seed(1)
    parameters=initialize(layer_dims)
    costs=[]
    for i in range(0,iter):
        Af,caches=l_model_forward(train_x,parameters,layer_dims)
        cost=compute_cost(Af,train_y)
        z11=cost[0]
        y11=z11[0]
        grads=l_model_backprop(caches,Af,train_y,layer_dims)
        parameters=optimize(grads,parameters,layer_dims,learning_rate)
        if i%100==0:
            print("cost after "+str(i)+" iterations is "+str(y11))
            costs.append(y11)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per100s)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

g=0
num_features=844*973*3
l=[]
for f in range(0,10):
    im = Image.open('dataset/final_dataset/potatoes_reshaped/new'+str(g)+'.jpg', 'r')
    pix_val = list(im.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    l.append(pix_val_flat)
g=0
for f in range(0,10):
    im = Image.open('dataset/final_dataset/carrots_reshaped/new'+str(g)+'.jpg', 'r')
    pix_val = list(im.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    l.append(pix_val_flat)
A=np.asarray(l)
B =np.reshape(A, (-1, num_features))
train_x=B.T 

h=[]
for i in range(0,10):
    h.append(1)
for j in range(0,10):
    h.append(0)

A1=np.asarray(h)
B1=np.reshape(A1, (-1, 20))
train_y=B1


layer_dims=[train_x.shape[0],21,7,5,1]
parameters=l_layer_nn(train_x,train_y,0.1,layer_dims,300)

print(type(parameters))

l1=[]
g=0
for f in range(0,2):
    im = Image.open('dataset/final_dataset/test_images_reshaped/new'+str(g)+'.jpg', 'r')
    pix_val = list(im.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    l1.append(pix_val_flat)
    g=g+1
A=np.asarray(l1)
B =np.reshape(A, (-1, num_features))
train_x1=B.T 

Arohan,caches=l_model_forward(train_x1,parameters,layer_dims)

for j in range(0,Arohan.shape[1]):
    if Arohan[0][j] >= 0.5:
        percent=100*Arohan[0][j]
        print("It is a potato with "+str(percent)+ " % "+"accuracy")
    else:
        percent=100-100*Arohan[0][j]
        print("It is a carrot with "+str(percent)+ " % "+ "accuracy")


