# import numpy as np
# import cv2
# import tensorflow as tf
# IMG_SIZE = 50
# model = tf.keras.models.load_model('64x3-CNN.model')
#
#
# img = cv2.resize(cv2.imread('/home/congdao/PycharmProjects/getstarted/machinelearningbasic/data/cat.jpg', cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
# TEST = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
# print(model.predict([TEST]))

import cv2
import tensorflow as tf
import os
import numpy as np

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value
TEST_DIR = '/home/congdao/PycharmProjects/getstarted/machinelearningbasic/data/'
test =[]
def prepare():
    IMG_SIZE = 50  # 50 in txt-based
    for img in os.listdir(TEST_DIR):
        # print(img)
        filepath = os.path.join(TEST_DIR, img)
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
        test.append([np.array(new_array), img])
    return test
model = tf.keras.models.load_model("cat_dog.model")
test = prepare()
# X_test =[i[0] for i in test]
# print(X_test)
for img in test:
    prediction = model.predict([img[0]])
    print(prediction, img[1])
    if prediction[0][0] > 0.5: print('dog')
    else: print('cat')
# print(prediction.shape)
# prediction = model.predict(X_test)
print(prediction)
# print(CATEGORIES[int(prediction[0][0])])