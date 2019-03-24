#-*- coding=utf-8 -*-
from __future__ import print_function
import datetime,os,sys
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# ================================================================================
# 设置参数

job_name = 'base'

train_data_path = 'zz_train_imdb'
valid_data_path = 'zz_test_imdb'
scatter_train_data_path = ''
scatter_valid_data_path = ''

outputdir  = './aa_network_result_' + job_name



max_words = 10000  # We will only consider the top 10,000 words in the dataset
maxlen = 100  # We will cut reviews after 100 words

embedding_dim = 8 

batch_size = 512  # Training and Testing batch_size  
epoch_num = 10

train_samples = 20000
test_samples = 5000


# ================================================================================
# 预处理文本


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('\n================================================================================ %s\n'%nowtime)
print('start preprocessing texts data...\n')

from tqdm import tqdm
def texts_gen():
    with open(train_data_path,'r') as f,tqdm(total = train_samples) as pbar:      
        while True:
            text = (f.readline().rstrip().split('\t')[-1]).replace('\004',' ')
            if not text:
                break
            if len(text) > maxlen:
                text = text[0:maxlen]
            pbar.update(1)
            yield text
            
texts = texts_gen()
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

#-----------------------------------------------------------------------------------

#def data_gen(data_file):
#    while True:
#        with open(data_file,'r') as f:
#            while True:
#                lines = [f.readline() for i in range(batch_size)]
#                if not lines[-1]:
#                    break    
#                labels = np.array([int(line.strip().split('\t')[-2]) for line in lines])
#                texts = [(line.strip().split('\t')[-1]).replace('\004',' ') for line in lines]
#                sequences = tokenizer.texts_to_sequences(texts)
#                datas = pad_sequences(sequences,maxlen)
#                yield datas,labels  
                

#train_gen = data_gen(train_data_path)
#test_gen = data_gen(valid_data_path)

#------------------------------------------------------------------------------------

# 将数据打散到一个文件一个样本

def scatter_data(data_file, scatter_data_path):
    if not os.path.exists(scatter_data_path):
        os.makedirs(scatter_data_path)
    for idx,line in enumerate(open(data_file,'r')):
        with open(scatter_data_path + str(idx) + '.txt','w') as f:
             f.write(line)

if not scatter_train_data_path or not scatter_valid_data_path:
    scatter_train_data_path = 'data/train/'
    scatter_valid_data_path = 'data/valid/'
    scatter_data(train_data_path,scatter_train_data_path)
    scatter_data(valid_data_path,scatter_valid_data_path)


# 定义Sequence数据管道， 可以多线程读数据

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,n_samples,data_path,batch_size=batch_size,shuffle=True):
        'Initialization'
        self.data_path = data_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        datas, labels = self.__data_generation(batch_indexes)
        return datas, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __read_file(self,file_name):
        with open(file_name) as f:
            line = f.readline()
        return line

    def __data_generation(self, batch_indexes):

        'Generates data containing batch_size samples'
        # Initialization
        lines = [self.__read_file(self.data_path + str(i) + '.txt') for i in batch_indexes]
        labels = np.array([int(line.strip().split('\t')[0]) for line in lines])
        texts = [line.strip().split('\t')[-1] for line in lines]
        sequences = tokenizer.texts_to_sequences(texts)
        datas = pad_sequences(sequences,maxlen)

        return datas,labels


train_gen = DataGenerator(train_samples,scatter_train_data_path)
valid_gen = DataGenerator(test_samples,scatter_valid_data_path)
                
# ================================================================================
# 自定义评估指标 auc 

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# ================================================================================
# 定义模型结构

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('\n================================================================================ %s\n'%nowtime)
print('start construct model ...\n')

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D ,Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Conv1D(32,5,activation = 'relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.summary()

n_gpus = len(K.tensorflow_backend._get_available_gpus())

# Replicates the model on 8 GPUs.
try:
    parallel_model = multi_gpu_model(model, gpus=n_gpus,cpu_relocation=True)
except:
    parallel_model = model

parallel_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[auc])

# ================================================================================
# 训练模型

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('\n================================================================================ %s\n'%nowtime)
print('start fit model ...\n')

steps_per_epoch = train_samples // batch_size
validation_steps = test_samples // batch_size

history = parallel_model.fit_generator(train_gen,
                         steps_per_epoch = steps_per_epoch,
                         epochs = epoch_num,
                         validation_data=valid_gen,
                         validation_steps = validation_steps,
                         use_multiprocessing=True,
                         workers=6
                         )

# ================================================================================

# 保存结果

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('\n================================================================================ %s\n'%nowtime)
print('save results...\n')

import os
import pandas as pd

if not os.path.exists(outputdir):
    os.makedirs(outputdir)


# 保存得分
auc = history.history['auc']
val_auc = history.history['val_auc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(auc) + 1)

dfhistory  = pd.DataFrame({'epoch':epochs,'train_loss':loss,'valid_loss':val_loss,
                  'train_auc':auc,'valid_auc':val_auc})

print(dfhistory)
dfhistory.to_csv(outputdir + '/metrics_result',sep = '\t',index = None)

# 保存模型
model.save(outputdir + '/model.h5')


########
#######
#####
####
###
##
#
