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

import numpy as np


def random_rotation(directory:str, class_list:list = ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp'], angle:float= 45.0, iter:int=3):
  """
  Function that rotates all images in a directory

  Args:
    directory (str): path directory to folder of subfolders of images
    class_list (list): list of class labels
    angle (float): maximum amount image can rotate, image will rotate from -angle to angle
    iter (int): determines how many new augmented data will be generated per image 
                Eg. iter = 3 means 3 new randomly roated image will be done per 1 input image
  """
  try:
    #doing data augmentation on every class
    for i in range(0,len(class_list)):
      class_name = class_list[i]
      #getting image count
      image_count = len(list(glob.glob(directory + class_name +'/*.jpg'))) #assumes that images are in the directory/class_name folder

      #running data augmentation, currently all files are named image_{imageNumber}_{rotationValue}
      for j in range(1,image_count+1):
        path = directory + class_name + f"/image_{str(j)}.jpg" #new filename
        img = cv2.imread(path) #read image
        img = rotation(img, angle, iter, path) #perform data augmentation

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return {'status': 'ok'}
  
  except Exception as e:
    return {'status': e}


def rotation(img:np.ndarray, angle:float, iter:int, path:str):
    """
    Function that generates iter number of random rotations of one image

    Args:
        img (np.ndarray): image read by openCV
        angle (float): maximum amount image can rotate, image will rotate from -angle to angle
        iter (int): determines how many new augmented data will be generated per image 
                    Eg. iter = 3 means 3 new randomly roated image will be done per 1 input image
        path (str): filepath to save rotated image
                    Images will be in the filepath of filename_1.jpg, filename_2.jpg etc..
    """
    i = 0
    if '.jpg' in filename or '.png' in filename:
      filename = path[:-4] + "_{}.jpg".format(i) #create unique filename
    else:
      filename = path + "_{}.jpg".format(i) #create unique filename

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

      


if __name__ == "__main__":
  directory = '/Data/'
  status = random_rotation(directory)
  print(status)