import numpy as np
import cv2
import csv
import colorsys

# Variables to change
videoFilename = 'videos/single_test.MOV'
bobRGB = np.array([71, 130, 174])
pivotLRGB = np.array([214, 163, 52])
pivotRRGB = np.array([90, 158, 107])
framerate = 60
dataFilename = 'data/singlepositions.csv'
videoOutputFilename = 'output.mp4'


# Color bounds in HSV
maxHSV = np.array([179, 255, 255])
minHSV = np.array([0, 0, 0])

bobHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(bobRGB/255.))), maxHSV)).astype(int)
bobLowerBound = np.maximum(bobHSV - 30, minHSV)
bobUpperBound = np.minimum(bobHSV + 30, maxHSV)

pivotLHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pivotLRGB/255.))), maxHSV)).astype(int)
pivotLLowerBound = np.maximum(pivotLHSV - 30, minHSV)
pivotLUpperBound = np.minimum(pivotLHSV + 30, maxHSV)

pivotRHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pivotRRGB/255.))), maxHSV)).astype(int)
pivotRLowerBound = np.maximum(pivotRHSV - 30, minHSV)
pivotRUpperBound = np.minimum(pivotRHSV + 30, maxHSV)


cap = cv2.VideoCapture(videoFilename)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1') # Define the codec
video = cv2.VideoWriter('output.mp4', fourcc, framerate, (width,height))
data = []

while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break

    # if cap.get(cv2.CAP_PROP_POS_FRAMES) > 30:
    #     break

    # Create masks for each colored marker
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bobMask = cv2.inRange(frameHSV, bobLowerBound, bobUpperBound)
    pivotLMask = cv2.inRange(frameHSV, pivotLLowerBound, pivotLUpperBound)
    pivotRMask = cv2.inRange(frameHSV, pivotRLowerBound, pivotRUpperBound)

    # Image filtering
    kernalOpen = np.ones((width//200,width//200))
    kernalClose = np.ones((width//100,width//100))
    bobMaskOpen = cv2.morphologyEx(bobMask,cv2.MORPH_OPEN, kernalOpen)
    bobMaskClose = cv2.morphologyEx(bobMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pivotLMaskOpen = cv2.morphologyEx(pivotLMask,cv2.MORPH_OPEN, kernalOpen)
    pivotLMaskClose = cv2.morphologyEx(pivotLMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pivotRMaskOpen = cv2.morphologyEx(pivotRMask,cv2.MORPH_OPEN, kernalOpen)
    pivotRMaskClose = cv2.morphologyEx(pivotRMaskOpen,cv2.MORPH_CLOSE, kernalClose)

    bobContours = cv2.findContours(bobMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pivotLContours = cv2.findContours(pivotLMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pivotRContours = cv2.findContours(pivotRMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bobMoments = cv2.moments(bobContours[0])
    pivotLMoments = cv2.moments(pivotLContours[0])
    pivotRMoments = cv2.moments(pivotRContours[0])

    # Calculate centroids
    bobCentroid = (int(bobMoments['m10']/bobMoments['m00']),int(bobMoments['m01']/bobMoments['m00']))
    pivotLCentroid = (int(pivotLMoments['m10']/pivotLMoments['m00']),int(pivotLMoments['m01']/pivotLMoments['m00']))
    pivotRCentroid = (int(pivotRMoments['m10']/pivotRMoments['m00']),int(pivotRMoments['m01']/pivotRMoments['m00']))

    # Create frame with marked centroids
    centroidim = frame.copy()
    cv2.circle(centroidim, bobCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotLCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotRCentroid, 3, (0,0,0), -1)
    cv2.drawContours(centroidim, bobContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, pivotLContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, pivotRContours[1], -1, (0,0,0), 2)

    # Display frame for review
    # cv2.namedWindow('review', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('review', width//2, height//2)

    # cv2.imshow("review", cv2.resize(frame, (0, 0), None, .5, .5))
    # cv2.waitKey(0)
    # # cv2.imshow("review", cv2.resize(pivotLMask, (0, 0), None, .5, .5))
    # # cv2.waitKey(0)
    # # cv2.imshow("review", cv2.resize(pivotLMaskOpen, (0, 0), None, .5, .5))
    # # cv2.waitKey(0)
    # # cv2.imshow("review", cv2.resize(pivotLMaskClose, (0, 0), None, .5, .5))
    # # cv2.waitKey(0)
    # cv2.imshow("review", cv2.resize(centroidim, (0, 0), None, .5, .5))
    # cv2.waitKey(1)

    # Write frame to video
    if ret == True:
        video.write(centroidim)
        print('.', end='', flush=True)

    # Append positions for this frame to the data
    time = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)/framerate
    data.append((time, *pivotLCentroid, *pivotRCentroid, *bobCentroid))


# Save the position data to a csv
with open(dataFilename, 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['t','pivotLx','pivotLy','pivotRx','pivotRy','bobx','boby'])
    for row in data:
        writer.writerow(row)


# Cleanup opencv
video.release()
cap.release()
cv2.destroyAllWindows()
