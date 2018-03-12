import numpy as np
import cv2
import csv
import colorsys

# Variables to change
videoFilename = 'videos/single.mov'
bobRGB = np.array([56, 134, 173])
pivotRGB = np.array([225, 163, 70])
framerate = 30
dataFilename = 'data/singlepositions.csv'
videoOutputFilename = 'output.mp4'


# Color bounds in HSV
maxHSV = np.array([179, 255, 255])
minHSV = np.array([0, 0, 0])

bobHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(bobRGB/255.))), maxHSV)).astype(int)
bobLowerBound = np.maximum(bobHSV - 40, minHSV)
bobUpperBound = np.minimum(bobHSV + 40, maxHSV)

pivotHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pivotRGB/255.))), maxHSV)).astype(int)
pivotLowerBound = np.maximum(pivotHSV - 40, minHSV)
pivotUpperBound = np.minimum(pivotHSV + 40, maxHSV)



cap = cv2.VideoCapture(videoFilename)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1') # Define the codec
video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))
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
    bobMask = cv2.inRange(frameHSV, bobLowerBound, bobUpperBound)
    pivotMask = cv2.inRange(frameHSV, pivotLowerBound, pivotUpperBound)

    # Image filtering
    kernalOpen = np.ones((width//100,width//100))
    kernalClose = np.ones((width//20,width//20))
    bobMaskOpen = cv2.morphologyEx(bobMask,cv2.MORPH_OPEN, kernalOpen)
    bobMaskClose = cv2.morphologyEx(bobMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pivotMaskOpen = cv2.morphologyEx(pivotMask,cv2.MORPH_OPEN, kernalOpen)
    pivotMaskClose = cv2.morphologyEx(pivotMaskOpen,cv2.MORPH_CLOSE, kernalClose)

    bobContours = cv2.findContours(bobMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pivotContours = cv2.findContours(pivotMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bobMoments = cv2.moments(bobContours[0])
    pivotMoments1 = cv2.moments(pivotContours[1][0])
    pivotMoments2 = cv2.moments(pivotContours[1][1])

    # Calculate centroids
    bobCentroid = (int(bobMoments['m10']/bobMoments['m00']),int(bobMoments['m01']/bobMoments['m00']))
    pivotCentroid1 = (int(pivotMoments1['m10']/pivotMoments1['m00']),int(pivotMoments1['m01']/pivotMoments1['m00']))
    pivotCentroid2 = (int(pivotMoments2['m10']/pivotMoments2['m00']),int(pivotMoments2['m01']/pivotMoments2['m00']))

    if pivotCentroid1[0] > pivotCentroid2[0]:
        pivotCentroidR = pivotCentroid1
        pivotCentroidL = pivotCentroid2
    else:
        pivotCentroidR = pivotCentroid2
        pivotCentroidL = pivotCentroid1

    # Create frame with marked centroids
    centroidim = frame.copy()
    cv2.circle(centroidim, bobCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotCentroid1, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotCentroid2, 3, (0,0,0), -1)
    cv2.drawContours(centroidim, bobContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, pivotContours[1], -1, (0,0,0), 2)

    # Display frame for review
    # cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('review', width//2, height//2)

    # cv2.imshow("review", cv2.resize(frame, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pivotMask, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pivotMaskOpen, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(pivotMaskClose, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(centroidim, (0, 0), None, .5, .5))
    # cv2.waitKey(1)

    # Write frame to video
    if ret == True:
        video.write(centroidim)
        print('.', end='', flush=True)

    # Append positions for this frame to the data
    time = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)/framerate
    data.append((time, *pivotCentroidL, *pivotCentroidR, *bobCentroid))


# Save the position data to a csv
with open(dataFilename, 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['t','pivotLx','pivotLy','pivotRx','pivotRy','bobx','boby'])
    for row in data:
        writer.writerow(row)


# cleanup opencv
video.release()
cap.release()
cv2.destroyAllWindows()
