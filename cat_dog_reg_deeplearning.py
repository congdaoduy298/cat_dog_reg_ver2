import os
import numpy as np
from random import shuffle
import cv2

def load_label(folder):
    if folder == 'Dog': return 1
    return 0
IMG_SIZE = 50
TRAIN_DIR = '/home/congdao/PetImages'
def create_train_data():
    train_data = []
    for folder in os.listdir(TRAIN_DIR):
        PATH = os.path.join(TRAIN_DIR, folder)
        print(PATH)
        label = load_label(folder)
        for num in os.listdir(PATH):
             path = os.path.join(PATH, num)
             try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                train_data.append([np.array(img), label])
             except Exception as e:
                 pass
    shuffle(train_data)
    return train_data

# create_train_data()
# np.save('train_data.npy', train_data)
train_data = np.load('train_data.npy', allow_pickle=True)
# print(len(train_data))

X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([i[1] for i in train_data])
for i in range(10):
    print(y[i])
# X = []
# y = []
# for features, label in train_data:
#     X.append(features)
#     y.append(label)
#
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# print(X[0])
# print(y[0])
import pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()