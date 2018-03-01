import numpy as np
import cv2
import csv
import colorsys

# Variables to change
filename = "antiphase_short.mov"
blueRGB = np.array([99, 178, 227])
orangeRGB = np.array([255, 198, 102])
pinkRGB = np.array([254, 123, 175])
framerate = 30


# Color bounds in HSV
maxHSV = np.array([179, 255, 255])
minHSV = np.array([0, 0, 0])

blueHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(blueRGB/255.))), maxHSV)).astype(int)
blueLowerBound = np.maximum(blueHSV - 30, minHSV)
blueUpperBound = np.minimum(blueHSV + 30, maxHSV)

orangeHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(orangeRGB/255.))), maxHSV)).astype(int)
orangeLowerBound = np.maximum(orangeHSV - 30, minHSV)
orangeUpperBound = np.minimum(orangeHSV + 30, maxHSV)

pinkHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pinkRGB/255.))), maxHSV)).astype(int)
pinkLowerBound = np.maximum(pinkHSV - 30, minHSV)
pinkUpperBound = np.minimum(pinkHSV + 30, maxHSV)


cap = cv2.VideoCapture(filename)
data = []

while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break
    height, width = frame.shape[:2]

    # Create masks for each colored marker
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blueMask = cv2.inRange(frameHSV, blueLowerBound, blueUpperBound)
    orangeMask = cv2.inRange(frameHSV, orangeLowerBound, orangeUpperBound)
    pinkMask = cv2.inRange(frameHSV, pinkLowerBound, pinkUpperBound)

    # Image filtering
    kernalOpen = np.ones((width//100,width//100))
    kernalClose = np.ones((width//10,width//10))
    blueMaskOpen = cv2.morphologyEx(blueMask,cv2.MORPH_OPEN, kernalOpen)
    blueMaskClose = cv2.morphologyEx(blueMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    orangeMaskOpen = cv2.morphologyEx(orangeMask,cv2.MORPH_OPEN, kernalOpen)
    orangeMaskClose = cv2.morphologyEx(orangeMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pinkMaskOpen = cv2.morphologyEx(pinkMask,cv2.MORPH_OPEN, kernalOpen)
    pinkMaskClose = cv2.morphologyEx(pinkMaskOpen,cv2.MORPH_CLOSE, kernalClose)

    blueContours = cv2.findContours(blueMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orangeContours = cv2.findContours(orangeMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pinkContours = cv2.findContours(pinkMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blueMoments = cv2.moments(blueContours[0])
    orangeMoments = cv2.moments(orangeContours[0])
    pinkMoments = cv2.moments(pinkContours[0])

    # Calculate centroids
    blueCentroid = (int(blueMoments['m10']/blueMoments['m00']),int(blueMoments['m01']/blueMoments['m00']))
    orangeCentroid = (int(orangeMoments['m10']/orangeMoments['m00']),int(orangeMoments['m01']/orangeMoments['m00']))
    pinkCentroid = (int(pinkMoments['m10']/pinkMoments['m00']),int(pinkMoments['m01']/pinkMoments['m00']))

    # Append positions for this frame to the data
    time = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)/framerate
    data.append((time, *blueCentroid, *orangeCentroid, *pinkCentroid))


    # Create frame with marked centroids
    centroidim = frame.copy()
    cv2.circle(centroidim, blueCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, orangeCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pinkCentroid, 3, (0,0,0), -1)

    # Display frame for review
    cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('review', width//2, height//2)

    # cv2.imshow("review", cv2.resize(frame, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pinkMask, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pinkMaskOpen, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pinkMaskClose, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    cv2.imshow("review", cv2.resize(centroidim, (0, 0), None, .5, .5))
    cv2.waitKey(1)


# Save the position data to a csv
with open('positions.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['t','bluex','bluey','orangex','orangey','pinkx','pinky'])
    for row in data:
        writer.writerow(row)

# cleanup opencv
cap.release()
cv2.destroyAllWindows()
