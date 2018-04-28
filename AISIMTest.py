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
"""
import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import optimizers
"""
from PIL import Image
from numpy import array
from keras.models import load_model
import numpy as np


#image parent relative folder paths and model relative folder path
modelDir = './models/'
testRelPath = './validation/0to17/'

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

#loads model and weights
model = load_model('90kperepolchwith30kmodel60acc.h5')
model.load_weights('./90kperepolchwith30kweights60acc.h5')
"""
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
#model.dropout(0.5)
model.add(Dense(256))

model.add(Dense(128))

#3 classes and softmax for multi-class activation
model.add(Dense(classes_num, activation='softmax'))

model.load_weights('./90kperepolchwith30kweights60acc.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=learningRate),
              metrics=['accuracy'])
"""

#directories and ten images for testing model
allDir = ['./outsidevalid/0to17/','./outsidevalid/18to60/','./outsidevalid/60plus/']
allImages = ['test1.jpg','test2.jpg','test3.jpg','test4.jpg','test5.jpg','test6.jpg',
             'test7.jpg','test8.jpg','test9.jpg','test10.jpg']


#loop through directories and images and predict based on loaded model
under18 = 0
middle = 0
plus = 0
cur = 0
for dirr in allDir:
    thingsToClassify = []
    for imag in allImages:
        thingsToClassify = []
        img = Image.open(dirr+imag)
        img = img.resize((150,150))
        thingsToClassify.append(np.asarray(img, dtype=np.uint8))
        arr = array(thingsToClassify)
        classes = model.predict_classes(arr,batch_size=16,verbose=0)
        print(str(dirr+imag),end=' ')
        for cl in classes:
            
            print(cl, " ")
            if(cur == 0):
                if(cl == 0):
                    under18 += 1
            if(cur == 1):
                if(cl == 2):
                    middle += 1
            if(cur == 2):
                if(cl == 1):
                    plus += 1


    arr = []
    cur+=1


#prints stats
print("0 to 17 age group had: ",under18,"/10")
print("18 to 60 age group had: ",middle,"/10")
print("60 plus age group had: ",plus,"/10")
print("Accuracy is overall showing at ",(under18+middle+plus),"/30 or ",((under18+middle+plus)/30.0)*100,"%")

