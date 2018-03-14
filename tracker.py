import numpy as np
import cv2
import csv
import colorsys

# Variables to change
videoFilename = "videos/antiphase.MP4"
bobLRGB = np.array([65, 113, 162])
bobRRGB = np.array([])
pivotLRGB = np.array([217, 175, 87])
pivotRRGB = np.array([198, 90, 124])
framerate = 60
dataFilename = 'data/singlepositions.csv'
videoOutputFilename = 'output.mp4'


# Color bounds in HSV
maxHSV = np.array([179, 255, 255])
minHSV = np.array([0, 0, 0])

bobLHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(bobLRGB/255.))), maxHSV)).astype(int)
bobLLowerBound = np.maximum(bobLHSV - 40, minHSV)
bobLUpperBound = np.minimum(bobLHSV + 40, maxHSV)

bobRHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(bobRRGB/255.))), maxHSV)).astype(int)
bobRLowerBound = np.maximum(bobRHSV - 40, minHSV)
bobRUpperBound = np.minimum(bobRHSV + 40, maxHSV)

pivotLHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pivotLRGB/255.))), maxHSV)).astype(int)
pivotLLowerBound = np.maximum(pivotLHSV - 40, minHSV)
pivotLUpperBound = np.minimum(pivotLHSV + 40, maxHSV)

pivotRHSV = np.rint(np.multiply(np.array(colorsys.rgb_to_hsv(*list(pivotRRGB/255.))), maxHSV)).astype(int)
pivotRLowerBound = np.maximum(pivotRHSV - 40, minHSV)
pivotRUpperBound = np.minimum(pivotRHSV + 40, maxHSV)


# Initialize capture, data recording, and writer
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
    bobLMask = cv2.inRange(frameHSV, bobLLowerBound, bobLUpperBound)
    bobRMask = cv2.inRange(frameHSV, bobRLowerBound, bobRUpperBound)
    pivotLMask = cv2.inRange(frameHSV, pivotLLowerBound, pivotLUpperBound)
    pivotRMask = cv2.inRange(frameHSV, pivotRLowerBound, pivotRUpperBound)

    # Image filtering
    kernalOpen = np.ones((width//100,width//100))
    kernalClose = np.ones((width//20,width//20))
    bobLMaskOpen = cv2.morphologyEx(bobLMask,cv2.MORPH_OPEN, kernalOpen)
    bobLMaskClose = cv2.morphologyEx(bobLMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    bobRMaskOpen = cv2.morphologyEx(bobRMask,cv2.MORPH_OPEN, kernalOpen)
    bobRMaskClose = cv2.morphologyEx(bobRMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pivotLMaskOpen = cv2.morphologyEx(pivotLMask,cv2.MORPH_OPEN, kernalOpen)
    pivotLMaskClose = cv2.morphologyEx(pivotLMaskOpen,cv2.MORPH_CLOSE, kernalClose)
    pivotRMaskOpen = cv2.morphologyEx(pivotRMask,cv2.MORPH_OPEN, kernalOpen)
    pivotRMaskClose = cv2.morphologyEx(pivotRMaskOpen,cv2.MORPH_CLOSE, kernalClose)

    bobLContours = cv2.findContours(bobLMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bobRContours = cv2.findContours(bobRMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pivotLContours = cv2.findContours(pivotLMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pivotRContours = cv2.findContours(pivotRMaskOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bobLMoments = cv2.moments(bobLContours[0])
    bobRMoments1 = cv2.moments(bobRContours[0])
    pivotLMoments = cv2.moments(pivotLContours[0])
    pivotRMoments = cv2.moments(pivotRContours[0])

    # Calculate centroids
    bobLCentroid = (int(bobLMoments['m10']/bobLMoments['m00']),int(bobLMoments['m01']/bobLMoments['m00']))
    bobRCentroid = (int(bobRMoments['m10']/bobRMoments['m00']),int(bobRMoments['m01']/bobRMoments['m00']))
    pivotLCentroid = (int(pivotLMoments['m10']/pivotLMoments['m00']),int(pivotLMoments['m01']/pivotLMoments['m00']))
    pivotRCentroid = (int(pivotRMoments['m10']/pivotRMoments['m00']),int(pivotRMoments['m01']/pivotRMoments['m00']))

    # Create frame with marked centroids
    centroidim = frame.copy()
    cv2.circle(centroidim, bobLCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, bobRCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotLCentroid, 3, (0,0,0), -1)
    cv2.circle(centroidim, pivotRCentroid, 3, (0,0,0), -1)
    cv2.drawContours(centroidim, bobLContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, bobRContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, pivotLContours[1], -1, (0,0,0), 2)
    cv2.drawContours(centroidim, pivotRContours[1], -1, (0,0,0), 2)

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

    # Write frames to video
    if ret == True:
        video.write(centroidim)
        print('.', end='', flush=True)

    # Append positions for this frame to the data
    time = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)/framerate
    data.append((time, *pivotCentroidL, *pivotCentroidR, *bobCentroidL, *bobCentroidR))


# Save the position data to a csv
with open('data/antiphasepositionsTEST.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['t','pivotLx','pivotLy', 'pivotRx', 'pivotRy','bobLx','bobLy','bobRx','bobRy'])
    for row in data:
        writer.writerow(row)

# Cleanup opencv
video.release()
cap.release()
cv2.destroyAllWindows()
