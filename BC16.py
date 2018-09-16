import csv
import  cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples=[]
path='./data3/'
with open(path+'driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples,train_size=0.1,test_size=0.05)
def trimmed(image):
    trimmed=image[70:135,0:320]
    return trimmed
def flipped(image,steering):
    flipped_image=cv2.flip(image,1)
    flipped_measurement=float(steering)*(-1.0)
    return (flipped_image,flipped_measurement)
def read(path):
    img=cv2.imread(path)
    image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples=shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name1,name2,name3 = path+'IMG/'+batch_sample[0].split('\\')[-1],path+'IMG/'+batch_sample[1].split('\\')[-1],path+'IMG/'+batch_sample[2].split('\\')[-1]
                center_image,left_image,right_image = read(name1),read(name2),read(name3)
                C=0.3 #correction
                center_angle,left_angle,right_angle = float(batch_sample[3]),float(batch_sample[3])+C,float(batch_sample[3])-C
                images.append(trimmed(center_image))
                images.append(trimmed(left_image))
                images.append(trimmed(right_image))
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#NN model(Nvidia based):
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

model=Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(65,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) #subsample=stride
model.add(Dropout(0.25))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(50,kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(10,kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,          nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h16')