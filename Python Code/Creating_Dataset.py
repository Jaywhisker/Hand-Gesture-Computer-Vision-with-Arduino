#Importing the libraries
import numpy as np
import cv2
import time
import os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains the function for
Creating Dataset:
collect_dataset(directory)

#The algorithm:

MASKING
- cap will connect to your web camera
- capture the first frame of the web camera (this will capture the background)
- crop the first frame according to the coordinates of the bounding box where the 
  camera will constantly be detecting for hand gestures
- greyscale the first frame and addd gaussian blur to remove any hard lines

- the webcam will endlessly run and constantly update its frame
- each frame will then be cropped according to the coordinates of the same bounding box, 
  followed by being grayscaled and gaussian blurred
- subtract the pixel values between the gray scaled new frame and first frame (background)
- if the pixel values are within the threshold values of (25-255), convert pixel value to
  be white and the rest will be converted to black
- this will apply the masking onto any object that was not in the background before 
  (which in this case will be the hands)

SAVING 
- the path of the dataset folder is set and the directory is redirected there
- every frame that is cropped and masked will be save under a unique name into 
  the directory
- this will create our dataset

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#Function to collect dataset
#Requires:directory contains Dataset folder path
#To break the function, just press hold and 'q' key
def collect_dataset(directory): 
  
  #Connecting the live webcam
  cap = cv2.VideoCapture(0)

  #Reading from the live webcam, ret returns True if the Webcam is working while first will 
  #returns the frame
  ret, first = cap.read() 

  # Save the first image as reference
  first_crop = first[100:324, 400:624] #Cropping the frame 
  first_gray = cv2.cvtColor(first_crop, cv2.COLOR_BGR2GRAY) #Converting to Grayscale
  first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0) #Gaussian blur to remove hardlines

  i = 0
  os.chdir(directory) #setting path to the directory

  while True:
      ret, frame = cap.read()

      if not ret:
          break

      cv2.rectangle(frame,(400,100), (624,324), (0,255,0), 2) #cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

      frame_rect = frame[100:324, 400:624] #cutting the image around the bounding box
      gray = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (21, 21), 0) 

      # In each iteration, calculate absolute difference between current frame and reference frame
      difference = cv2.absdiff(gray, first_gray)

      # Apply thresholding to eliminate noise
      thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.dilate(thresh, None, iterations=2)

      cv2.imshow("frame", frame)
      cv2.imshow("thresh", thresh)

      filename = "frame_01_{name}.jpg".format(name = str(i)) #create a unique file name
      cv2.imwrite(filename, thresh) #save the file
      i += 1

      key = cv2.waitKey(3) & 0xFF
      # if the `q` key is pressed, break from the lop
      if key == ord("q"):
          break
