import numpy as np
import cv2


cap = 'Fast Driving Car On Straight Road.mp4'

class Video():

	def open_video(self, cap):

		cap = cv2.VideoCapture(cap)

		while(cap.isOpened()):
		    ret, frame = cap.read()

		    h = frame.shape[0]
		    w = frame.shape[1]

		    orb = cv2.ORB_create()

		    kp = orb.detect(frame,None)

		    kp, des = orb.compute(frame, kp)

		    img3 = np.zeros((h, w, 3), np.uint8)
		    img2 = cv2.drawKeypoints(img3, kp, None, color=(0,255,0), flags=0)
		    #img2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
		    img2 = cv2.resize(img2, (w // 2, h // 2))

		    #print(frame)

		    points = cv2.KeyPoint_convert(kp)
		    x = points[0]
		    y = points[1]
		    print(x[0], y[0])

		    img3 = cv2.line(img3, (x[0], y[0]), (x[1], y[1]), color = (0, 255, 0))

		    cv2.imshow('frame',img2)
		    if cv2.waitKey(1) & 0xFF == ord('k'):
		        break

		cap.release()
		cv2.destroyAllWindows()

vid = Video()

vid.open_video(cap)