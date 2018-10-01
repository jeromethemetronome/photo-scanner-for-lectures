# USAGE
# python scan.py --image images/page.jpg

# import packages
from pyimagesearch.transform import four_point_transform
import numpy as np
import cv2
from skimage.filters import threshold_local
import imutils

from matplotlib import pyplot as plt

# load the image from the rep

image = cv2.imread('P1000229.jpg',0)
    
#Resize image to make it faster
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


# grayscale, blur, and find edges
blur = cv2.GaussianBlur(image,(5,5),0)
gray = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow('P1000229.jpg', image)

# show the original image and the edge detected image
print("We detect edges")
cv2.imshow("Blur", blur)
cv2.imshow("Edges", edged)


cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

#Analyze the contours
for c in cnts:
	# approximate the contours
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour of the piece of paper
cv2.drawContours(image, [screenCnt], -1, (250, 0, 0), 4)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

'''
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
'''

# show the original and scanned images
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)

cv2.imwrite('P1000269_new.jpg', imutils.resize(warped, height = 650))