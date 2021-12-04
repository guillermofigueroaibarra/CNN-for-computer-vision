'''
Name: Guillermo Figueroa
Description: this program will predict what a picture contains whether is a animal, object etc. 

Source Code: josephlee94
Source website: https://github.com/josephlee94/intuitive-deep-learning/tree/master/Part%202:%20Image%20Recognition%20CIFAR-10

'''


'''
Name: Guillermo Figueroa
Description: this program will predict what a picture contains whether is a animal, object etc. 

Source Code: josephlee94
Source website: https://github.com/josephlee94/intuitive-deep-learning/tree/master/Part%202:%20Image%20Recognition%20CIFAR-10

'''



# EXPLORING AND PROCESSING THE DATA
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print('x_train shape:', x_train.shape)


print('y_train shape:', y_train.shape)


print(x_train[0])



##import matplotlib.pyplot as plt
##%matplotlib inline
##img = plt.imshow(x_train[0])


print('The label is:', y_train[0])


img = plt.imshow(x_train[1])


print('The label is:', y_train[1])



import keras
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)
print('The one hot label is:', y_train_one_hot[1])



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
x_train[0]


# BUILDING AND TRAINING OUR CONVOLUTIONAL NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))


model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


model.summary()






model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




hist = model.fit(x_train, y_train_one_hot, 
           batch_size=32, epochs=20, 
           validation_split=0.2)




plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()



model.evaluate(x_test, y_test_one_hot)[1]



model.save('my_cifar10_model.h5')




# TESTING OUT WITH YOUR IMAGES

my_image = plt.imread("cat.jpg")



from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3))

img = plt.imshow(my_image_resized)


import numpy as np
probabilities = model.predict(np.array( [my_image_resized,] ))

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])


