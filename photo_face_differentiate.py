'''
CNN : Image Processing
RNN and LSTM : Stock Prices or Large no no Column or numerical data


CNN : Kersa 
relu
max Pooling
Flatten

terms
Dense
Dropout

kersa,tensorflow,theano :
'''



# import keras library 
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import keras
#initializing the CNN
classifier = Sequential()
#step-1
classifier.add(Conv2D(32, (3,3), input_shape=(64, 64, 3), activation="relu"))
#step2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#adding a second convolution layer:
classifier.add(Conv2D(32, (3,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#STEP-3:FLATTENING
classifier.add(Flatten())
#step-4:Full connection                                                                                                     
classifier.add(Dense(units = 128,activation = "relu"))
classifier.add(Dense(units = 1,activation = "sigmoid")) #if only two characteristics in images to differentiate then use "sigmoid" and if more than two characteristic in images than use softmax

#compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#compiling the CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range= 0.2, horizontal_flip = True)#rescale = 1./255'''scaling each image''', shear_range = 0.2, zoom_range= 0.2, horizontal_flip = True
test_datagen = ImageDataGenerator(rescale = 1./225)

training_set = train_datagen.flow_from_directory(r"C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\training_set",target_size = (64, 64),batch_size = 32, class_mode = 'binary')#target_size = (64, 64)'''convert every img to 64*64''',batch_size = 32'''how many imgs to work on at once''', class_mode = 'binary'
training_set.class_indices

test_set=test_datagen.flow_from_directory(r"C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\test_set",target_size = (64, 64),batch_size = 32, class_mode = 'binary')
test_set.class_indices


classifier.fit_generator(training_set, steps_per_epoch = 300, epochs=5, validation_data = test_set, validation_steps = 50)#training_set, steps_per_epoch = 100'''how many images to run''', epochs=2'''how many times number of program is run''', validation_data = test_set, validation_steps = 50
classifier.save_weights(r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\dogs_cats_model.h5')

classifier.load_weights(r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\dogs_cats_model.h5')
#part 3-making new predictions
import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r"C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\test_img.jpg",target_size=(64,64))
test_image
test_image=image.img_to_array(test_image) #take pixel show image like get or imread in opencv  
test_image
test_image=np.expand_dims(test_image,axis=0) # img pixel is stored in row
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
    print(prediction)
else:
    prediction='cat'
    print(prediction)

100:1model
200:2model












