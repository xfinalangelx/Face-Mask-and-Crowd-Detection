# import packages
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# the default parameters
DATASET_DIR = "dataset"
PLOT_IMG_NAME = "accuracy_and_loss_curve.png"
MODEL_NAME = "mask_detector.model"
INIT_LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

def load_data(dataset_dir):

	imagePaths = list(paths.list_images(dataset_dir))
	data = []
	labels = []

	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)
		data.append(image)
		labels.append(label)
	data = np.array(data, dtype="float32")
	labels = np.array(labels)
	return data, labels

def one_hot_encode(labels):
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	labels = to_categorical(labels)

	return labels, lb

def build_model(learning_rate, epoch):
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	model = Model(inputs=baseModel.input, outputs=headModel)

	for layer in baseModel.layers:
		layer.trainable = False

	opt = Adam(learning_rate=learning_rate, decay=learning_rate / epoch)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def plot_acc_and_loss_graph(epochs, History, plot_img_name):
	
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, epochs), History.history["loss"], label="Train Loss")
	plt.plot(np.arange(0, epochs), History.history["val_loss"], label="Validation Loss")
	plt.plot(np.arange(0, epochs), History.history["accuracy"], label="Train Accuracy")
	plt.plot(np.arange(0, epochs), History.history["val_accuracy"], label="Validation Accuracy")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_img_name)

data, labels = load_data(DATASET_DIR)
labels, lb = one_hot_encode(labels)

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

model = build_model(INIT_LEARNING_RATE, EPOCHS)
History = model.fit(
	aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
	steps_per_epoch=int(len(x_train) / BATCH_SIZE),
	validation_data=(x_val, y_val),
	validation_steps=int(len(x_val) / BATCH_SIZE),
	epochs=EPOCHS)

predIdxs = model.predict(x_test, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
testIdxs = np.argmax(y_test, axis=1)
accuracy = round((np.sum(predIdxs == testIdxs) / len(predIdxs)) * 100, 2)
print(f"Accuracy: {accuracy}%")
print(classification_report(y_test.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
model.save(MODEL_NAME, save_format="h5")
plot_acc_and_loss_graph(EPOCHS, History, PLOT_IMG_NAME)