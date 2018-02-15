import numpy as np
import cv2

# filename = "img_%02d.jpg"
filename = "test.mov"

cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
    ret, frame = cap.read()

    height, width = frame.shape[:2]

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([33,80,40])
    upperBound = np.array([102,255,255])

    mask = cv2.inRange(frameHSV, lowerBound, upperBound)

    # Image filtering
    kernalOpen = np.ones((width//70,width//70))
    kernalClose = np.ones((width//10,width//10))
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernalOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE, kernalClose)

    cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('review', width//2, height//2)

    cv2.imshow("review", frame)
    cv2.waitKey(0)
    cv2.imshow("review", mask)
    cv2.waitKey(0)

    cv2.imshow("review", maskClose)
    cv2.waitKey(0)
    cv2.imshow("review", maskOpen)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
