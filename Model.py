"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#This code file contains the function for

Defining the model:
creating_model(class_names)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#import dependencies
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential


def creating_model(class_names):
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
  #this layer will determine the model prediction

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
