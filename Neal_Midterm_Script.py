## Brendan Neal
## Directory ID: bneal12
## ENPM673 Midterm

##------------Importing Libraries---------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import random

##----------------------------QUESTION 2-------------------------------------##

original_video = cv.VideoCapture('ball.mov') #Get Video
count = 0 #Start Frame Count

if (original_video.isOpened() == False):
    print("Error Opening File!")

# While Video is successfully loaded:
while(original_video.isOpened()):
    count = count + 1 #Increase Counter
    success, img = original_video.read() #Read Image
    if success: #If Successfully Read Image:
        GrayScale = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #Set Image to GrayScale
        GaussBlur = cv.GaussianBlur(GrayScale, (7,7), 0, 0) #Using the Gaussian Blur with a 7x7 kernel to blur the image before processing.
        #Find Circles using HoughCircles(). Gradient Detection method, 1 pixel resolution, 20 indices between circles, 150 upper threshold for canny
        # 15 threshold for center detection. The ball is roughly 11 pixels, so I search for balls between 1 and 15 just to be sure.
        detected_circles = cv.HoughCircles(GaussBlur,cv.HOUGH_GRADIENT,1,20,param1=150,param2=15,minRadius=1,maxRadius=15) 
        if detected_circles is not None: #If the ball is found
            detected_circles = np.uint16(np.around(detected_circles)) #Convert Data Type
            for circle in detected_circles[0,:]: #For Every Detected Circle (should be 1)
                cv.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),2) # Outline Ball in Green
        cv.imshow("Question 2 -- Ball Detection", img )  #Display Black and White Thresholded Image
        cv.waitKey(10)
    else:
        original_video.release() #release video
        break #break out of loop


##----------------------------QUESTION 3-------------------------------------##

Original_Train_Track_Image = cv.imread('train_track.jpg') #Read Image
Original_Train_Track_Image_Copy = cv.imread('train_track.jpg') #Copy Image to Draw on.

cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Original Image", 500, 400)
cv.imshow("Original Image", Original_Train_Track_Image); cv.waitKey(0) #Show Original Image

points_of_interest = ((1483,1020), (1541,1020), (950,1920), (2050,1920)) #Define my Points of Interest ("Corners" on the Railroad)

for points in points_of_interest: #For Each Point of Interest
    cv.circle(Original_Train_Track_Image_Copy, points, 10, (0,0,255), -1) #Plot the Point on the Image

cv.namedWindow("Points of Interest", cv.WINDOW_NORMAL)
cv.resizeWindow("Points of Interest", 500, 400)
cv.imshow("Points of Interest", Original_Train_Track_Image_Copy); cv.waitKey(0) #Display what I am warping

Points_to_Transform_To = ((0,0), (58,0), (0,900), (58,900)) #Define the points (in a flat plane) that I want to warp the image to.

Array_Points1 = np.array(points_of_interest, np.float32) #Convert to Numpy Array
Array_Points2 = np.array(Points_to_Transform_To, np.float32) #Convert to Numpy Array

TransformationMatrix = cv.getPerspectiveTransform(Array_Points1, Array_Points2) #Get the Transformation Matrix
Top_Down_Image = cv.warpPerspective(Original_Train_Track_Image, TransformationMatrix, (58,900)) #Warp the Perspective

cv.namedWindow("Top Down Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Top Down Image", 500, 400)
cv.imshow("Top Down Image", Top_Down_Image); cv.waitKey(0) #Display the Warped Image

GrayScale_Tracks = cv.cvtColor(Top_Down_Image, cv.COLOR_RGB2GRAY) #Set Image to GrayScale
cv.namedWindow("Gray Top Down Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Gray Top Down Image", 500, 400)
cv.imshow("Gray Top Down Image", GrayScale_Tracks); cv.waitKey(0) #Display Gray Scale Image

GaussBlur_Tracks = cv.GaussianBlur(GrayScale_Tracks, (7,7), 0, 0) #Use the Gaussian Blur with a 7x7 kernel to blur the image before processing.
cv.namedWindow("Blur Gray Top Down Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Blur Gray Top Down Image", 500, 400)
cv.imshow("Blur Gray Top Down Image", GaussBlur_Tracks); cv.waitKey(0) #Display the Blurred Image

EdgeDetection_Tracks = cv.Canny(GaussBlur_Tracks, 50, 100) #Using Canny edge detection with lower threshold of 50 and upper threshold of 100 to find edges.
cv.namedWindow("Edge Detect Top Down Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Edge Detect Top Down Image", 500, 400)
cv.imshow("Edge Detect Top Down Image", EdgeDetection_Tracks); cv.waitKey(0) #Display Edge deteced Image

#Use Houg Lines to detect lines. I want at least 300 votes in the accumulator to detect a line
DetectedLines = cv.HoughLines(EdgeDetection_Tracks, 1, np.pi / 180, 300, None, 0, 0) 


Line_Info = np.zeros((3,1)) #Initialize Line info.

if DetectedLines is not None:
    for line in range(0, len(DetectedLines)):
        D = DetectedLines[line][0][0] #Grab Rho from a Line
        theta = DetectedLines[line][0][1] #Grab Theta from a Line
        Aeq = math.cos(theta) #Get A
        Beq = math.sin(theta) #Get B
        X_0 = Aeq * D #Get X_0 where Line Exists
        Y_0 = Beq * D #Get Y_0 where Line Exists
        FirstPoint = (int(X_0 + 1000*(-Beq)), int(Y_0 + 1000*(Aeq))) #Initialize Start Line Point by Expanding X_0 and Y_0
        Line_Info[line-1][0] = FirstPoint[0] #Append the X data to Line info for later
        SecondPoint = (int(X_0 - 1000*(-Beq)), int(Y_0 - 1000*(Aeq))) #Initialize End Line Point by Expaning X_0 and Y_0
        cv.line(Top_Down_Image, FirstPoint, SecondPoint, (255,0,255), 3) #Plot the Detected Lines on Image in Magenta
cv.namedWindow("Track Lines", cv.WINDOW_NORMAL)
cv.resizeWindow("Track Lines", 500, 400)
cv.imshow("Track Lines", Top_Down_Image); cv.waitKey(0) #Display Detected Lines


# Solve for the average distance between tracks by taking the longest X distance between detected lines. I detect some extra lines of the same rho and theta
#but at different points. Inspecting the Image shows that the furthest line is the line on the track itself.
CurrentMin = Line_Info[0]
CurrentMaxDist = 0
for i in range(len(Line_Info)):
    if (Line_Info[i] < CurrentMin):
        CurrentMin = Line_Info[i]
    elif (Line_Info[i] - CurrentMin > CurrentMaxDist):
        CurrentMaxDist = Line_Info[i] - CurrentMin
print("The Average Distance Between the Train Tracks in Pixels is:", CurrentMaxDist)

##----------------------------QUESTION 4-------------------------------------##
Original_Balloon_Image = cv.imread('hotairbaloon.jpg') #Load the Image
Original_Balloon_Image_Copy = cv.imread('hotairbaloon.jpg') #Load a Copy to Draw On

GrayScale_Balloon = cv.cvtColor(Original_Balloon_Image, cv.COLOR_RGB2GRAY) #Set Image to GrayScale
cv.namedWindow("GreyScale Balloons", cv.WINDOW_NORMAL)
cv.resizeWindow("GreyScale Balloons", 500, 400)
cv.imshow("GreyScale Balloons", GrayScale_Balloon); cv.waitKey(0) #Show The Grayscale Image

Blur_Balloon = cv.GaussianBlur(GrayScale_Balloon, (11,11), 0, 0) #Use Gaussian Blur to blur the Image before Processing
cv.namedWindow("Blurred Balloons", cv.WINDOW_NORMAL)
cv.resizeWindow("Blurred Balloons", 500, 400)
cv.imshow("Blurred Balloons", Blur_Balloon); cv.waitKey(0) #Display the Blurred Image

_, Threshold_Balloon = cv.threshold(Blur_Balloon, 95, 255, cv.THRESH_BINARY_INV) #Threshold the Image to Binary with a limit of 95
cv.namedWindow("Thresholded Balloons", cv.WINDOW_NORMAL)
cv.resizeWindow("Thresholded Balloons", 500, 400)
cv.imshow("Thresholded Balloons", Threshold_Balloon); cv.waitKey(0) #Display the Thresolded Image

#The thresholding gives me too many holes between balloons, so I use morphology to close the image.
Closed_Balloon = cv.morphologyEx(Threshold_Balloon, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(25,25)))
cv.namedWindow("Closed Balloons", cv.WINDOW_NORMAL)
cv.resizeWindow("Closed Balloons", 500, 400)
cv.imshow("Closed Balloons", Closed_Balloon); cv.waitKey(0) #Display the new thresholded balloons

NumLabels, Label_IDXs, Stats, Centroid_Component = cv.connectedComponentsWithStats(Closed_Balloon) #Find Connected Components with the statistics in an Image

for i in range(1, Label_IDXs.max()): #For Each labeled connected component. Index 0 is the whole image for some reason.
    xloc, yloc, width, height, area = Stats[i] #Grab the stats for the connected component
    #Generate a random color for labeling (ensures different colors)
    for i in range(3):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
    #As a final check (should only be balloons in the image) but only analyze objects with an area greater than 100 pixels^2
    if area > 100:
        cv.rectangle(Original_Balloon_Image_Copy, (xloc,yloc), (xloc+width, yloc+height), (r,g,b), 3) #Draw a "bounding box" around the detected objects

cv.namedWindow("Identified Balloons", cv.WINDOW_NORMAL)
cv.resizeWindow("Identified Balloons", 500, 400)
cv.imshow("Identified Balloons", Original_Balloon_Image_Copy); cv.waitKey(0) #Display the Balloons with the bounding boxes of different colors.






