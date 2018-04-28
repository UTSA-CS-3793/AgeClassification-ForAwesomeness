# -*- coding: utf-8 -*-
"""
Team AI SIM Training/Model Generator file
Tutorials Used/referenced
https://github.com/gsadhas/Age-Classification-using-CNN/blob/master/CNNArc.py
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
"""

import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import optimizers



#image parent relative folder paths and model relative folder path
trainRelPath = './images/completetrain'
validRelPath = './images/completetest'
modelDir = './models/'

#image,batch size, sample per epoch, steps and other variables
epochs = 40
imgW, imgH = 150, 150
batchSz = 16
samplesPerEpoch = 90000 #dependss on images in folders
validationSteps = 800
convolutionSize = 3
poolSz = 3
classes_num = 3
learningRate = 0.0001


#sequential model
model = Sequential()

#six convolution layers image width height and rgb
model.add(Convolution2D(128, (convolutionSize, convolutionSize), input_shape=(imgW, imgH, 3)))
model.add(Activation("relu"))
model.add(Convolution2D(128, (convolutionSize, convolutionSize), input_shape=(imgW, imgH, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(poolSz, poolSz)))

model.add(Convolution2D(64, (convolutionSize, convolutionSize)))
model.add(Activation("relu"))
model.add(Convolution2D(64, (convolutionSize, convolutionSize)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(poolSz, poolSz)))

model.add(Convolution2D(32, (convolutionSize, convolutionSize)))
model.add(Activation("relu"))
model.add(Convolution2D(32, (convolutionSize, convolutionSize)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(poolSz, poolSz)))


#flatten out data
model.add(Flatten())


#Fully Connected Networks
model.add(Dense(512))

model.add(Activation("relu"))

model.add(Dense(256))

model.add(Dense(128))

#3 classes and softmax for multi-class activation
model.add(Dense(classes_num, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=learningRate),
              metrics=['accuracy'])

#manipulates the image
#rescales, rotates, flips etc
train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#test is only resized
test = ImageDataGenerator(rescale=1. / 255)


#pull training data
trainData = train.flow_from_directory(
    trainRelPath,
    target_size=(imgH, imgW),
    batch_size=batchSz,
    class_mode='categorical')


#pull validation data
validData = test.flow_from_directory(
    validRelPath,
    target_size=(imgH, imgW),
    batch_size=batchSz,
    class_mode='categorical')


#train model
model.fit_generator(
    trainData,
    samples_per_epoch=samplesPerEpoch,
    epochs=epochs,
    validation_data=validData,
    validation_steps=validationSteps)


#ensure path for saving model exists or make it
if not os.path.exists(modelDir):
  os.mkdir(modelDir)
  
#save model and weights
  
model.save('./models/45ktotalmodel.h5')
model.save_weights('./models/45ktotalkweights.h5')