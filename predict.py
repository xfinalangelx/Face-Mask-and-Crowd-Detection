# Import packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# default parameters
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

def predict(face_props, image, model):

	(x, y, w, h) = face_props
 
	face = image[y:y+h, x:x+w]

	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

	face = cv2.resize(face, (224, 224))

	face = img_to_array(face)

	face = preprocess_input(face)

	face = np.expand_dims(face, axis=0)

	pred = model.predict(face)

	return pred
	


def detect_faces(image, face_cascade):

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray_image, minNeighbors=6, minSize=[30,30])

	return faces

def label_face(face):	

	(x,y,w,h) = face

	[[wear_mask, not_wear_mask]] = predict(face, image, maskNet)

	if wear_mask > not_wear_mask:
		label = "Wearing Mask"
		label_color = GREEN_COLOR
	else:
		label = "Not Wearing Mask"
		label_color = RED_COLOR

	cv2.putText(image, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 2)

	cv2.rectangle(image, (x,y), (x+w, y+h), label_color, 2)

maskNet = load_model("mask_detector.model")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

images = [
	cv2.imread("dataset/with_mask/8-with-mask.jpg"),
	cv2.imread("dataset/without_mask/8.jpg"),
	cv2.imread("img_one.jpeg"),
	cv2.imread("img_two.jpeg.jpg")
]

counts = 0

for image in images:
	faces = detect_faces(image, face_cascade)

	for face in faces:
		label_face(face)
		counts = len(faces)
		if counts > 1:
			cv2.putText(image, "ALERT! CROWD DETECTED", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

	cv2.imshow("Face Mask and Crowd Detector", image)

	cv2.waitKey(0)
