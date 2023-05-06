"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains functions for

Data Augmentation:
random_rotation(class_list, directory, angle, iter):
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#import dependencies
import cv2
import random
import pathlib
import glob

#rotation function that includes how many rotation per image and where to save it
def rotation(img, angle, iter, path):
    i = 0
    filename = path[:-4] + "{}.jpg".format(i) #create unique filename
    
    for num in range(iter): #how many images per image
      angle = int(random.uniform(-angle, angle)) #get a random angle
			
			#rotate the image
      h, w = img.shape[:2]
      M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
      img = cv2.warpAffine(img, M, (w, h))
      
			#show image and save file
      cv2.imshow(img)
      cv2.imwrite(filename, img)
      i+= 1

      
def random_rotation(class_list = ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp'], directory, angle, iter):

  #doing data augmentation on every class
  for i in range(0,len(class_list)):
    class_name = class_list[i]
    #getting image count
    image_count = len(list(glob.glob(directory + class_name +'/*.jpg')))

    #running data augmentation, currently all files are named 1 (1).jpg, 1 (2).jpg etc
    for j in range(1,image_count+1):
      path = directory + class_name + '/1 (' + str(j) +").jpg" #new filename
      img = cv2.imread(path) #read image
      img = rotation(img, angle, iter, path) #perform data augmentation

  cv2.waitKey(0)
  cv2.destroyAllWindows()
