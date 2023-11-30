"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains the function for

Accessing Firebase:
Firebase_setup(secretspath, url)

Testing the model:
Hand_gesture_recognition(secretspath, url, class_names, modelpath):
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#importing dependencies
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras


def setup_firebase(secretspath:str, url:str):
  """
  Connect to firebase 

  Args:
      secretspath (str): filepath to secrets.json file
      url (str): filepath to firebase url
  """
  #fetch the service account key JSON file contents
  cred = credentials.Certificate(secretspath)

  #initialize the app with a service account, granting admin privileges
  firebase_admin.initialize_app(cred, {
      'databaseURL': url #database URL 
  })

  #checking if led database exist, if it doesn't create one with default value 0
  ref = db.reference("/led")
  if ref.get() == None:
    db.reference("/").set({"led":0})
    
    

def recognise_hand_gesture(firebase:bool, secretspath:str, url:str, modelpath:str, class_names = ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp']):
  """
  Function to recognise hand gesture live and update the firebase DB based on results
  Function runs indefinitely, stop function by press and hold q
  Args:
      firebase (bool): option to connect to firebase, if False, just do evaluation
      secretspath (str): filepath to secrets.json file
      url (str): filepath to firebase url
      modelpath (str): filepath to model 
      class_names (list, optional): list of class labels. Defaults to ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp'].
  """
  if firebase:
    setup_firebase(secretspath, url)
  
  #start streaming video from webcam 
  video_frame = cv2.VideoCapture(0)

  #reading from live webcam
  ret, first = video_frame.read()

  #getting the background image for masking 
  first_crop = first[100:324, 400:624] 
  first_gray = cv2.cvtColor(first_crop, cv2.COLOR_BGR2GRAY)
  first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

  #loading the Computer Vision model
  reconstructed_model = keras.models.load_model(modelpath) #path to model

  #while the webcam is running
  while True:
      ret, frame = video_frame.read()
      if not ret: #if the webcam stop running, break the for code 
          break

      #drawing the box on top of the live webcam
      #allows user to know where to the hand gestures will be recognised
      cv2.rectangle(frame,(400,100), (624,324), (0,255,0), 2)

      #get the frame from the live webcam and crop the image according to bounding box
      frame_rect = frame[100:324, 400:624] #cutting the image around the bounding box

      #create the masked image for model to predict (same code creating dataset)
      gray = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (21, 21), 0) 
      difference = cv2.absdiff(gray, first_gray)
      thresh = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.dilate(thresh, None, iterations=2)
      cv2.imshow("frame", frame)

      #convert image into data for the model to predict
      img_array = tf.keras.utils.img_to_array(thresh)
      img_array = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
      img_array = tf.expand_dims(img_array, 0) #Create a batch

      #run the model to get the predictions 
      predictions = reconstructed_model.predict(img_array)
      score = tf.nn.softmax(predictions[0])

      print(
          "This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score))
      )

      if firebase:
        #updating Firebase data based on the model prediction
        if np.argmax(score) == 0: #if high five, set 1
          ref = db.reference('/led') 
          ref.set(1)

        elif np.argmax(score) == 1: #if fist, set 0
          ref = db.reference('/led')
          ref.set(0)
        #print(db.reference('/led').get())

      time.sleep(0.5)
      
      key = cv2.waitKey(3) & 0xFF
      # if the `q` key is pressed, break from the lop
      if key == ord("q"):
          break

if __name__ == "__main__":
  secrets_path = ''
  firebase_url = ''
  model_path = ''
  recognise_hand_gesture(secrets_path, firebase_url, model_path)
