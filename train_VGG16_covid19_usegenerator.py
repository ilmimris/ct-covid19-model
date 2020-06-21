# USAGE
# python train_VGG16_covid19.py --dataset dataset

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-t", "--testpath", required=True,
	help="path to input testpath")
ap.add_argument("-p", "--plot", type=str, default="plot_VGG16.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19_VGG16.model",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
DATASET_PATH=args["dataset"]
TEST_PATH=args["testpath"]
BATCH_SIZE = 8
IMAGE_SIZE=(224,224)

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
# Train datagen here is a preprocessor
train_datagen = ImageDataGenerator(rescale=1./255,
 rotation_range=50,
 featurewise_center = True,
 featurewise_std_normalization = True,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.25,
 zoom_range=0.1,
 zca_whitening = True,
 channel_shift_range = 20,
 horizontal_flip = True ,
 vertical_flip = True ,
 validation_split = 0.2,
 fill_mode='constant')

train_batches = train_datagen.flow_from_directory(DATASET_PATH,
 target_size=IMAGE_SIZE,
 shuffle=True,
 batch_size=BATCH_SIZE,
 subset = 'training',
 seed=42,
 class_mode='binary',
 )

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,
 target_size=IMAGE_SIZE,
 shuffle=True,
 batch_size=BATCH_SIZE,
 subset = 'validation',
 seed=42,
 class_mode='binary',
 )

# Test datagen here is a preprocessor
test_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = test_datagen.flow_from_directory(
 TEST_PATH,
 target_size=IMAGE_SIZE,
 batch_size=1,
 shuffle=False,
 seed=42,
 class_mode='binary')

eval_generator.reset()

# Callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# load the VGG16 network, ensuring the head FC layer sets are left
# off
conv_base  = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

conv_base.trainable = False

# construct the head of the model that will be placed on top of the
# the base model
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
	train_batches,
	steps_per_epoch=train_batches.n // BATCH_SIZE,
	validation_data=valid_batches,
	validation_steps=valid_batches.n // BATCH_SIZE,
	epochs=EPOCHS,
	callbacks=[callbacks])

# make predictions on the testing set
print("[INFO] evaluating network...")
x = model.evaluate_generator(eval_generator,
 steps = np.ceil(len(eval_generator) / BATCH_SIZE),
 use_multiprocessing = False,
 verbose = 1,
 workers=1
 )
print('Test loss:' , x[0])
print('Test accuracy:', x[1])
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(eval_generator, steps = len(eval_generator.filenames))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(f'eval_generator.classes:{eval_generator.classes[0]}, y_pred:{y_pred[0]}')
cm = confusion_matrix(eval_generator.classes, y_pred)
print(cm)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('Classification Report')
target_names = ['Covid', 'Normal']
print(classification_report(eval_generator.classes, y_pred, target_names=target_names))

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = len(H.history["loss"])
plt.style.use("ggplot")
plt.figure(0)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# plot result side by side
eval_generator.reset() 
pred = model.predict_generator(eval_generator,64,verbose=1)
print("Predictions finished")

for index, probability in enumerate(pred):
	print(eval_generator.filenames[index])
	plt.figure(index+1)
	image_path = TEST_PATH + "/" +eval_generator.filenames[index]
	image = mpimg.imread(image_path)
	#BGR TO RGB conversion using CV2
	try:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	except Exception as e:
		pass

	pixels = np.array(image)
	plt.imshow(pixels)

	if probability > 0.5:
		plt.title("%.2f" % (probability[0]*100) + "% Normal")
		plt.savefig('pred/'+eval_generator.filenames[index])
	else:
		plt.title("%.2f" % ((1-probability[0])*100) + "% COVID19 Pneumonia")
		plt.savefig('pred/'+eval_generator.filenames[index])


# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")