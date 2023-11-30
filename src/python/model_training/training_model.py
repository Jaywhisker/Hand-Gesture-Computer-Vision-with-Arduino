"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains functions for

Defining the model:
creating_model(class_names)

Training Model:
preparing_dataset(filepath, train_val_split, seed)
visualise_dataset(train_ds)
train_model(class_names, train_ds, val_ds, batch_size, epochs, modelpath)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Importing dependencies 
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential


def preparing_dataset(filepath, train_val_split = 0.2, seed = 123):
  """
  Function to generate dataset for training (training and validation dataset)

  Args:
      filepath (string): filepath to the parent directory that contains subfolders of images (subfolder name is class name)
      train_val_split (float, optional): ratio for the train to validation dataset size split. Defaults to 0.2.
      seed (int, optional): splitting dataset seed. Defaults to 123.

  Returns:
      class_names: list of class labels for the dataset
      train_ds: training dataset
      val_ds: validation dataset
  """

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



def visualise_dataset(train_ds):
  """
  Helper function for visualising dataset
  Will show 9 images from the first batch of the train dataset

  Args:
      train_ds: training dataset
  """
  plt.figure(figsize=(10, 10))
  class_names = train_ds.class_names
      
  for images, labels in train_ds.take(1): #take first batch
    for i in range(9): #first 9 images
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")

      
def creating_model(class_names:list):
  """
  Create VGG16 model

  Args:
      class_names (list): list of class labels of the dataset

  Returns:
      model: trainable model architecture of vgg16
  """

  #loading VGG16 model
  #not using any of imagenet weights due to a custom data, remove the top predictor layer
  base_model = VGG16(weights= None, include_top=False, input_shape=(224,224,3)) 
  base_model.trainable = True # trainable weights

  #creating our own layers to add on to VGG16
  flatten_layer = layers.Flatten() #flatten outputs from VGG16
  dense_layer_1 = layers.Dense(50, activation='relu') #dense layer
  dense_layer_2 = layers.Dense(20, activation='relu') #dense layer
  prediction_layer = layers.Dense(len(class_names), activation='softmax') 
  #prediction layer, the number of outputs is the number of classes we have

  #merging the layers together to create our Computer Vision model
  model = models.Sequential([
      base_model, #VGG16
      flatten_layer,
      dense_layer_1,
      dense_layer_2,
      prediction_layer
  ])
  model.summary()
  return model


def train_model(class_names:list, train_ds, val_ds, modelpath:str, batch_size:int=32, epochs:int=15):
  """
  Function to train the vgg16 model

  Args:
      class_names (list): list of class labels of the dataset
      train_ds: training dataset
      val_ds: validation dataset
      modelpath (str): path where model will be saved
      batch_size (int, optional): batch size of dataset used for training. Defaults to 32.
      epochs (int, optional): number of epochs / iterations to run. Defaults to 15.
  """
  #creating model
  model = creating_model(class_names)
  
  #creating callback checkpoint to save the weights of the model with the least loss
  checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
  
  #training the model
  model.fit(train_ds, validation_data = val_ds, batch_size=batch_size, callbacks=[checkpoint], epochs=epochs)
  
  #save the best model
  model.save(modelpath)


if __name__ == "__main__":
  filepath = '/Data/'
  class_names, train_ds, val_ds = preparing_dataset(filepath)
  train_model(class_names, train_ds, val_ds)

