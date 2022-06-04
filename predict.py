hide_img = False 

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from model import create_model
import numpy as np
import operator
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img_path = "dataset/split_data/train/c0/img_34.jpg"
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
    
model = create_model()
model.load_weights("_weights.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

target_size=(150,150)

image = load_img(img_path, target_size=target_size)

image_arr = img_to_array(image) # convert from PIL Image to NumPy array

image_arr = np.expand_dims(image_arr, axis=0)
image_arr /= 255

predictions = model.predict(image_arr)

decoded_predictions = dict(zip(class_labels, predictions[0]))

decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

print()
count = 1
for key, value in decoded_predictions[:1]:
    word_split = key.split("_")
    if len(word_split) == 3:
        print("The person in the photo is", word_split[0], word_split[1], word_split[2])
    elif len(word_split) == 4:
        print("The person in the photo is", word_split[0], word_split[1], word_split[2], word_split[3])
    else:
        print("The person in the photo is", word_split[0], word_split[1])

if not hide_img:
    plt.imshow(image)
    plt.axis('off')
    plt.show()
