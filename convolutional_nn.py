import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = 'cats-vs-dogs-cnn-64x2-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

X = X/255.0

# print(NAME)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, validation_split=0.3, epochs=10, callbacks=[tensorboard])
# model.save('cat_dog.model')
img = cv2.resize(cv2.imread('/home/congdao/PycharmProjects/getstarted/machinelearningbasic/data/21.jpg', cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
TEST = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(model.predict(TEST))