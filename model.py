import os
import csv
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    #print (samples)
	
def img_bright(img):
    rand = np.random.uniform(0.3, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * rand
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img2
	
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for camera in range(3):
                    name = './data/IMG/'+batch_sample[camera].split('/')[-1]
                    #print (name)
                    image = cv2.imread(name)
                    #image = cv2.resize(image[60:140,:], (64,64))
                    #print (image.shape)
                    #image = img_bright(image)
                    img = image[60:140,:,:]
                    image = cv2.resize(img,(64, 64), interpolation=cv2.INTER_AREA)
                    #img = img[None, :, :, :]
                    #img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                  #  image = image/255.-0.5
                    #image = img_bright(img)
                    #print (name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    images.append(cv2.flip(image, 1))
                
                #angles = aappend_angle(angles, angle)
                angles.append(angle)
                angles.append(angle*-1.0)
                angles.append(angle + 0.2)
                angles.append((angle + 0.2)*-1.0)
                angles.append((angle - 0.2))
                angles.append((angle - 0.2)*-1.0)
                                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 64, 64, 3 # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
#model.add(Cropping2D(cropping=((50, 20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.50))
model.add(Dense(50))
model.add(Dropout(0.50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


#optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer='adam')
data = model.fit_generator(train_generator, samples_per_epoch=len(train_samples*6), validation_data=validation_generator, nb_val_samples=len(validation_samples*6), nb_epoch=5, verbose = 1)

model.save('model.h5')

print ("Model saved.")

plt.plot(data.history['loss'])
plt.plot(data.history['val_loss'])
plt.title('Loss summary')
plt.ylabel('Loss')
plt.xlabel('No. of EpochS')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()