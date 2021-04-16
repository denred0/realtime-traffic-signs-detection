import numpy as np
import cv2

# load cascade file
plate_cascade = cv2.CascadeClassifier('haar-cascades/haarcascade_russian_plate_number.xml')

def detect_and_blur_plate(img):

	plate_img = img.copy()
	roi = img.copy()

	# you should tune scaleFactor and minNeighbors for your purpose
	plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.4, minNeighbors=9)


	for (x, y, w, h) in plate_rects:
		roi = roi[y:y + h, x:x + w]
		
		# check if roi not empty
		if roi.shape[0] != 0 and roi.shape[1] != 0:
			blurred_roi = cv2.medianBlur(roi, 9)
            
			plate_img[y:y + h, x:x + w] = blurred_roi

	return plate_img
