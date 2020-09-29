from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random, os, glob
import cv2

# init params
epochs = 100
lr = 1e-3
batchSize = 64
imgDims = (96,96,3)

#create two lists
data = []
labels = []

#load images from the dataset (list comprhension)
imageFiles = [f for f in glob.glob(r'C:\Users\sydar_000\PycharmProjects\pythonProject1\dataset' + "/**/*",recursive=True
                                   )if not os.path.isdir(f)]
#shuffle the images
random.shuffle(imageFiles)

# convert images and label the categories
for img in imageFiles:

    # read, resize and appent to data after converting to array
    image = cv2.imread(img)
    image = cv2.resize(image, (imgDims[0], imgDims[1]))
    image = img_to_array(image)
    data.append(image)


    label = img.split(os.path.sep)[-2] #-2 means the folder name in the path
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label]) # [[1], [1], [0], ......]


#pre-processing
data = np.array(data, dtype="float") / 255 #every pixel have 0-255 vals
labels = np.array(labels)

#split dataset 0.2 for validation and 0.8 for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

#convert labels to categorical, 2 since man and woman
trainY = to_categorical(trainY, num_classes=2)# [[1,0],[0,1],[0,1],...]
testY = to_categorical(testY, num_classes=2)

#augmenting dataset (editing images)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1,shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    changeDim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        changeDim = 1

    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first",
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=changeDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))#reduce noise
    model.add(Dropout(0.25))#becomes overfit if not dropped

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=changeDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=changeDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=changeDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=changeDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())#converts 2D to 1D
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


# build model
model = build(width=imgDims[0], height=imgDims[1], depth=imgDims[2],
              classes=2)

# compile model
opt = Adam(lr = lr, decay = lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train model
T = model.fit_generator(aug.flow(trainX, trainY, batch_size=batchSize),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batchSize,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save('genderDetection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), T.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), T.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), T.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), T.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')