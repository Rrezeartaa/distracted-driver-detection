import argparse

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from model import create_model

model = create_model()
model.load_weights("_weights.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16
test_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            rescale=1./255, 
            zoom_range=0.1, 
            horizontal_flip=False,
            fill_mode='nearest') 

test_generator = test_datagen.flow_from_directory(
        '../../../dataset/split_data/test/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

score, acc = model.evaluate_generator(test_generator, len(test_generator.filenames))

print("score: ", score)
print("accuracy:", acc)
