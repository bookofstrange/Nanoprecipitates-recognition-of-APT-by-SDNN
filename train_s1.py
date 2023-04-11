# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:34:46 2023

@author: JiWei Yu
"""

#%% laod data
import numpy as np
import random
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense,Flatten,Input,Dropout
#%%
def data_batch(start,end,random_sample=False,num=5):
    if random_sample==False:
        data_train,label_train=np.load(f'feature/{start}.npy'),np.load(f'label/{start}.npy')
        for i in range(start+1,end):
            # k=ramdom.randint(0,400)
            feature,label=np.load(f'feature/{i}.npy'),np.load(f'label/{i}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    else:
        k=random.randint(start,end-1)
        data_train,label_train=np.load(f'feature/{k}.npy'),np.load(f'label/{k}.npy')
        for i in range(num-1):
            k=random.randint(start,end-1)
            feature,label=np.load(f'feature/{k}.npy'),np.load(f'label/{k}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    return data_train,label_train
data_train,label_train=data_batch(100,150,False,num=50)
data_val,label_val=data_batch(45,50,False,num=5)
#%% compile

class SimpleDense(keras.layers.Layer):
    def __init__(self,units=1,activation=None,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.activation=activation
    def build(self,input_shape):
        input_dim=input_shape[-1]
        self.W=self.add_weight(shape=(input_dim,self.units),initializer='random_normal',name='W')
        self.b=self.add_weight(shape=(self.units,),initializer='zeros',name='b')
    def call(self,inputs):
        y=tf.matmul(inputs,self.W)+self.b
        if self.activation is not None:
            y=self.activation(y)
        return y
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
        })
        return config
x=Input(shape=(128,5))
h=SimpleDense(2,tf.nn.relu)(x)
h=SimpleDense(1,tf.nn.relu)(h)
f=Flatten()(h)
f=Dropout(0.2)(f)
d=Dense(64,activation=tf.nn.relu)(f)
d=Dense(16,activation=tf.nn.relu)(d)
d=Dropout(0.2)(d)
y=Dense(1,tf.nn.sigmoid)(d)
#%%
model=keras.Model(inputs=x,outputs=y)
import os
checkpoint_save_path='./checkpoint/SDNN_V1.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------load the model-------')
    model.load_weights(checkpoint_save_path)
callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=True,
                                                save_best_only=False,
                                              )
#%%
# model=keras.models.load_model('SimpleDenseNN.keras',custom_objects={'SimpleDense':SimpleDense})
# model=keras.Model(inputs=x,outputs=y)
#%%
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics='accuracy')
ep=100
# history=model.fit(data_train,label_train,epochs=ep,batch_size=1024
#             ,validation_data=(data_val,label_val),callbacks=callback
#             )
# loss=tf.keras.losses.cross_entropy
model.summary()

#%%
'''
test_acc=[]
for i in range(60,65):
    # i='p4'
    data_test=np.load(f'feature/{i}.npy')
    p=model.predict((data_test),batch_size=1024)>0.5
    # p=model.predict((data_test),batch_size=1024)
    np.save(f'prediction/{i}.npy',p)
    # test_label=np.load(f'label/{i}.npy')
    # num=model.evaluate(data_test,test_label)
    # test_acc.append(num)
# test_acc=np.array(test_acc)
# np.save('数据处理/acc_SDNN.npy',test_acc[:,1])
# average_test_acc=np.sum(test_acc[:,1])/4
# print(average_test_acc)
'''
#%% component
'''
i='p0'
data_test=np.load(f'feature/{i}.npy')
data_test[:,:,0:3]=0
p=model.predict((data_test),batch_size=1024)>0.5
np.save(f'prediction_component/{i}.npy',p)
test_label=np.load(f'label/{i}.npy')
num=model.evaluate(data_test,test_label)
'''
#%% structure
'''
i=1
data_test=np.load(f'feature/{i}.npy')
data_test[:,:,3]=0
p=model.predict((data_test),batch_size=1024)
np.save(f'prediction_structure/{i}.npy',p)
test_label=np.load(f'label/{i}.npy')
num=model.evaluate(data_test,test_label)
'''
#%%
import matplotlib.pyplot as plt
h_dict=history.history
loss=h_dict['loss']
np.save('information/loss.npy',loss)
ep=[i for i in range(ep)]
val_loss=h_dict['val_loss']
plt.figure(dpi=200)
plt.plot(ep,loss,'go',label='Training loss')
plt.plot(ep,val_loss,'g--',label='Validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('information/loss.png')
plt.figure(dpi=200)
acc = h_dict["accuracy"]
np.save('information/acc.npy',acc)
val_acc = h_dict["val_accuracy"]
plt.plot(ep, acc, "bo", label="Training acc")
plt.plot(ep, val_acc, "b--", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
# plt.title(f'average_test_acc:{average_test_acc}')
plt.legend()
plt.savefig('information/acc.png')
#%%
# model.save('SimpleDenseNN.keras')
