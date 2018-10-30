import numpy as np
import cv2
from matplotlib import pyplot as plt

path = 'carDriveThing.jpg'

class Detect():

	def detector(self, path):

		img = cv2.imread(path,1)
		# Initiate ORB detector
		orb = cv2.ORB_create()
		# find the keypoints with ORB
		kp = orb.detect(img,None)
		# compute the descriptors with ORB
		kp, des = orb.compute(img, kp)

		img3 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

		# draw only keypoints location,not size and orientation

		img2 = cv2.drawKeypoints(img3, kp, None, color=(0,255,0), flags=0)

		plt.imshow(img2), plt.show()


de = Detect()

de.detector('carDriveThing.jpg')