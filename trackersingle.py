import numpy as np
import cv2
import csv
import colorsys

# Variables to change
filename = "single.mov"
blueRGB = np.array([56, 134, 173])
orangeRGB = np.array([225, 163, 70])
framerate = 30


# Color bounds in HSV
maxHSV = np.array([179, 255, 255])
minHSV = np.array([0, 0, 0])

blueHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(blueRGB/255.))), maxHSV)).astype(int)
blueLowerBound = np.maximum(blueHSV - 40, minHSV)
blueUpperBound = np.minimum(blueHSV + 40, maxHSV)

orangeHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(orangeRGB/255.))), maxHSV)).astype(int)
orangeLowerBound = np.maximum(orangeHSV - 40, minHSV)
orangeUpperBound = np.minimum(orangeHSV + 40, maxHSV)



cap = cv2.VideoCapture(filename)
data = []

while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break
    height, width = frame.shape[:2]

    # if cap.get(cv2.CAP_PROP_POS_FRAMES) > 30:
    #     break

    # Create masks for each colored marker
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blueMask = cv2.inRange(frameHSV, blueLowerBound, blueUpperBound)
    orangeMask = cv2.inRange(frameHSV, orangeLowerBound, orangeUpperBound)

    # Image filtering
    kernalOpen = np.ones((width//100,width//100))
    kernalClose = np.ones((width//20,width//20))
    blueMaskOpen = cv2.morphologyEx(blueMask,cv2.MORPH_OPEN, kernalOpen)
    blueMaskClose = cv2.morphologyEx(blueMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    orangeMaskOpen = cv2.morphologyEx(orangeMask,cv2.MORPH_OPEN, kernalOpen)
    orangeMaskClose = cv2.morphologyEx(orangeMaskOpen,cv2.MORPH_CLOSE, kernalClose)

    blueContours = cv2.findContours(blueMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orangeContours = cv2.findContours(orangeMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blueMoments = cv2.moments(blueContours[0])
    orangeMoments1 = cv2.moments(orangeContours[1][0])
    orangeMoments2 = cv2.moments(orangeContours[1][1])

    # Calculate centroids
    blueCentroid = (int(blueMoments['m10']/blueMoments['m00']),int(blueMoments['m01']/blueMoments['m00']))
    orangeCentroid1 = (int(orangeMoments1['m10']/orangeMoments1['m00']),int(orangeMoments1['m01']/orangeMoments1['m00']))
    orangeCentroid2 = (int(orangeMoments2['m10']/orangeMoments2['m00']),int(orangeMoments2['m01']/orangeMoments2['m00']))

    if orangeCentroid1[0] > orangeCentroid2[0]:
        orangeCentroidR = orangeCentroid1
        orangeCentroidL = orangeCentroid2
    else:
        orangeCentroidR = orangeCentroid2
        orangeCentroidL = orangeCentroid1

    # Create frame with marked centroids
    centroidim = frame.copy()
    cv2.circle(centroidim, blueCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, orangeCentroid1, 3, (0,0,0), -1)
    cv2.circle(centroidim, orangeCentroid2, 3, (0,0,0), -1)
    cv2.drawContours(centroidim, blueContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, orangeContours[1], -1, (0,0,0), 2)

    # Display frame for review
    cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('review', width//2, height//2)

    # cv2.imshow("review", cv2.resize(frame, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(orangeMask, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(orangeMaskOpen, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(orangeMaskClose, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    cv2.imshow("review", cv2.resize(centroidim, (0, 0), None, .5, .5))
    cv2.waitKey(1)

    # Append positions for this frame to the data
    time = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)/framerate
    data.append((time, *orangeCentroidL, *orangeCentroidR, *blueCentroid))


# Save the position data to a csv
with open('singlepositions.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['t','orangeLx','orangeLy','orangeRx','orangeRy','bluex','bluey'])
    for row in data:
        writer.writerow(row)

# cleanup opencv
cap.release()
cv2.destroyAllWindows()
