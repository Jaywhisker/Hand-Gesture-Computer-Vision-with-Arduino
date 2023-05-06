"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains functions for

Training Model:
preparing_dataset(filepath, train_val_split, seed)
visualise_dataset(train_ds)
train_model(train_ds, val_ds, model, batch_size, epochs, filepath)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Importing dependencies 
import tensorflow as tf
from tensorflow import keras
import pathlib
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from Model import model

#Function that splits the images into their training and validation dataset
#Requires: filepath to dataset, train_val_split size <1, currently default to 0.2, seed for randomly creating train and val ds, currently default to 123
#Returns class_name, train_dataset, val_dataset
def preparing_dataset(filepath, train_val_split = 0.2, seed = 123):

  #Getting the path to the dataset
  data_dir = pathlib.Path(filepath)

  #Checking if can access dataset as well as how much data is there in total
  image_count = len(list(data_dir.glob('*/*.jpg')))
  print("Total datasize:" + image_count)

  #variables for our data, batch size of 32 is a standard value
  #image size of 224 x 224 is based on the requirements of the VGG16 model we will be using
  batch_size = 32
  img_height = 224
  img_width = 224

  #splitting the dataset into train and validation data set
  #using a validation_split of 0.2 splits the data into 80% training and 20% validation
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=train_val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=train_val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  #to check the validation of our datatset by checking the labels of classes in our dataset
  class_names = train_ds.class_names
  
  return class_names, train_ds, val_ds



#Function to visualise dataset
#Requires: training dataset
#Returns: Image of dataset 
def visualise_dataset(train_ds):
  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1): #take first batch
    for i in range(9): #first 9 images
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")

      

def train_model(train_ds, val_ds, model, batch_size=32, epochs=15, filepath):
  #creating callback checkpoint to save the weights of the model with the least loss
  checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
  
  #training the model
  model.fit(train_ds,validation_data = val_ds, batch_size=batch_size, callbacks=[checkpoint], epochs=epochs)
  
  #save the best model
  model.save(filepath)
