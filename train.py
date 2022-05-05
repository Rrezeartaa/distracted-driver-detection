from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from model import create_model
from math import ceil

model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 40

train_datagen = ImageDataGenerator(
            rotation_range=10, 
            rescale=1./255, 
            zoom_range=0.1, 
            horizontal_flip=False,
            fill_mode='nearest') 

val_datagen = ImageDataGenerator(
            rotation_range=10,
            rescale=1./255, 
            zoom_range=0.1, 
            horizontal_flip=False,
            fill_mode='nearest') 

train_generator = train_datagen.flow_from_directory(
        'dataset/split_data/train/', 
        target_size=(150, 150),  
        batch_size=batch_size,
        class_mode='categorical')  

val_generator = val_datagen.flow_from_directory(
        'dataset/split_data/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

filepath="weights.h5"

checkpoint_callback = ModelCheckpoint(
                        filepath,
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stop_callback = EarlyStopping(
                monitor='val_acc',     
                patience=3,
                mode='max') 

callbacks_list = [checkpoint_callback, early_stop_callback]

history = model.fit_generator(
            train_generator,
            steps_per_epoch=(ceil(len(train_generator.filenames) // batch_size)),
            epochs=50,
            validation_data=val_generator,
            validation_steps=(ceil(len(val_generator.filenames) // batch_size)),
            callbacks=callbacks_list)
