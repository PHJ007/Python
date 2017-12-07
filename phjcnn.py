from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import backend as k
k.set_image_dim_ordering('tf')

model=Sequential()
model.add(Convolution2D(96,(11,11),strides=(4,4),input_shape=(227,227,3)
                        ,padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Convolution2D(256,(5,5),strides=(1,1),padding='same',
                        activation='relu',kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Convolution2D(384,(3,3),strides=(1,1),padding='same',
                        activation='relu',kernel_initializer='uniform'))
model.add(Convolution2D(384,(3,3),strides=(1,1),padding='same',
                        activation='relu',kernel_initializer='uniform'))
model.add(Convolution2D(256,(5,5),strides=(1,1),padding='same',
                        activation='relu',kernel_initializer='uniform'))

model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='sgd',
              metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,
                         data_format='channels_last')

test_datagen=ImageDataGenerator(rescale=1./255,data_format='channels_last')

train_generator=train_datagen.flow_from_directory('/home/phj/PycharmProjects/CNN_keras-master/core/train',
                                     target_size=(227,227),
                                     batch_size=50,
                                     class_mode='categorical')
validation_generator=test_datagen.flow_from_directory('/home/phj/PycharmProjects/CNN_keras-master/core/test',
                                     target_size=(227,227),
                                     batch_size=20,
                                     class_mode='categorical')
# print(train_generator.n)
model.fit_generator(train_generator,
                    steps_per_epoch=20,
                    epochs=500,
                    validation_data=validation_generator,
                    validation_steps=10,
                    verbose=1)