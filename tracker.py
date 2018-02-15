import numpy as np
import cv2
import csv

# filename = "img_%02d.jpg"
filename = "test.mov"

cap = cv2.VideoCapture(filename)
# positions = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),2))
positions = []

while(cap.isOpened()):
    ret, frame = cap.read()
    # print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    if(frame is None):
        break
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

    contours = cv2.findContours(maskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)

    centroid = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
    positions.append(centroid)
    # centroidim = maskOpen.copy()
    centroidim = frame.copy()
    cv2.circle(centroidim, centroid, 10, (0,0,0), -1)

    cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('review', width//2, height//2)

    cv2.imshow("review", frame)
    cv2.waitKey(10)
    # cv2.imshow("review", mask)
    # cv2.waitKey(0)
    #
    # cv2.imshow("review", maskClose)
    # cv2.waitKey(0)
    cv2.imshow("review", centroidim)
    cv2.waitKey(10)

# print(positions)
with open('positions.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['x','y'])
    for row in positions:
        writer.writerow(row)

cap.release()
cv2.destroyAllWindows()
