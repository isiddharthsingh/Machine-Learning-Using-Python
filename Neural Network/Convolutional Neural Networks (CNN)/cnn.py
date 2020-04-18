#Convolutional Neural Network


#Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initializing the CNN
classifier = Sequential()


#Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Adding second Convolutional Layer to increase test set Accuracy
classifier.add(Convolution2D(32,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Flattening
classifier.add(Flatten())


# Full Connection
#Hidden layer
classifier.add(Dense(units=128,activation='relu'))
#Output Layer
classifier.add(Dense(units=1,activation='sigmoid'))


#Compiling CNN 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting CNN to the Images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64 ),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)