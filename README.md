# Hand Gesture Computer Vision with Arduino

This project contains the code to create a simple hand gesture computer vision model. The hand gestures then can be used to control an Arduino MKR Wifi through a Firebase's real time database.

Link to in-depth step by step guide: https://www.notion.so/Smart-Home-With-Arduino-and-Computer-Vision-628f8458261f4b9cb6ba959f22b34212

Note: The blog uses the code that accomodates for running on Google Colab. There will 2 version of code in this GitHub, one that corresponds to the Google Colab code (tweaked to allow access to webcam) as well as a version that allows you to run the code locally if you have a GPU.

## Overview of Project

<div align = "center">
  <img src="https://user-images.githubusercontent.com/51687528/236611878-5cd06a73-8268-463c-8e0f-48cd8a831a2a.png" width = 70%>
</div>

This project is broken down into 3 main part: Data Collection, Training the model, Linking the Arduino to the Firebase

### Output: 
<div float="left" align="center">
    <img src = "https://user-images.githubusercontent.com/51687528/236612154-597cce93-70c1-4e6e-a608-725a8dc7a04e.png" width=35%>
    <img src="https://user-images.githubusercontent.com/51687528/236612165-170014ca-31f3-47da-ba07-4b76884ed85d.png" width=35%>
</div>
<p align="center">
  <em> LED switches off when a fist is detected and LED switches on when a high five is detected </em>
</p>

-----
## Dependencies
For this project, install the following libraries with pip
```
pip install numpy
pip install opencv-python
pip install tensorflow
pip install keras
pip install matplotlib
pip install firebase-admin
```
-----
### Collecting the Dataset

## Data Collection

As we only want to recognise hand gestures, we do not want other dependencies such as the background to affect our data. Thus, a specific bounding box is drawn and masking will be applied to that bounding box to only collect the hand gestures.

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236613036-564f7e4f-fe48-4514-bbcd-1b77055c9c73.png" width = 26.6%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613042-f64e14f6-b373-4070-95a2-bb4e08aa80dd.png" width = 20%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613047-eb44ce45-c2dd-424c-8c85-90bf65a833f9.png" width = 20%>
</div>
<p align="center">
  <em>First frame saves the background before it gets cropped according to the bounding box and grayscaled. This is our base background. </em>
</p>

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236613053-7f7eab1e-8f1a-4e93-bb4f-959fc8956a1c.png" width= 25.3%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613060-e95910aa-60ca-425e-a4e3-2a513c7c9560.png" width = 19%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613066-597745b0-354d-4c12-a7e0-3c30968cb698.png" width = 19%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613075-b5860883-fc68-4c1d-9755-53cda0f87c60.png" width = 19%>
</div>

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236613087-b7c75384-05b4-4b15-bfbb-6a01bc94f624.png" width= 25.3%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613090-daf17fdc-06d4-4b27-a58a-87cca4f40a3a.png" width = 19%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613095-c66c4c12-7df0-4068-9966-167d7ca4345f.png" width = 19%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613099-a839e042-2c8d-412b-bbb5-1951552cdd7c.png" width = 19%>
</div>

<p align="center">
  <em>Examples of how the masking works. The base background is masked, leaving just the hand gestures.</em>
</p>

### Function
`collect_dataset(directory)`: Function that takes in the directory to the folder that will save the dataset. Each folder should only contain ONE hand gesture. 

```
# How to use code
from Creating_Dataset import *

directory = "./Dataset/01_Highfive"
collect_dataset(directory)
```

Example of output:
<div align = "center">
  <img src="https://user-images.githubusercontent.com/51687528/236613141-40a4fda0-af6a-4414-bb4d-4e64573c8ff7.png" width = 70%>
</div>
<div>
  <p>      </p>
</div>

Repeat this for the other hand gestures:

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236613156-785007fc-13e2-4523-95b0-e26e9a61e0a8.png" width= 12%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613167-ec0c103c-e00a-4fc6-a98f-ba7096f4cd15.png" width = 12%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613176-fd2caa49-7a00-4138-81db-e82de1a82919.png" width = 12%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613194-83ee0517-8c6e-480a-8585-8cbeade864eb.png" width = 12%>
  <img src = "https://user-images.githubusercontent.com/51687528/236615804-fa9d6186-e0aa-4155-bd8e-d50fb39fdc4b.png" width = 12%>
  <img src = "https://user-images.githubusercontent.com/51687528/236613206-faadfb14-9750-4381-abdd-36f5d09f805d.png" width = 12%>
</div>
<p align="center">
  <em>Examples of dataset</em>
</p>

This will be your dataset for training your model. It is advised to look through the dataset and do some cleaning manually before moving on to the next step.
-----
## Data Augmentation

Data augmentation is important as it increases our model accuracy while also increasing our dataset size. I have chosen random rotation of -90° to 90° for our data augmentation as it covers the range where the hand direction could be in.

### Function
`random_rotation(class_list, directory,angle, iter)`: Function that takes in the list of class names, the directory that holds the folder of datasets, the angle that the image can rotate max and the number of data augmentation per image.


```
# How to use code
from Data_Augmentation import *

class_list is default as  ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp']
directory = "./Dataset"
angle is default as 45
iter is default as 3
random_rotation(class_list, directory, angle, iter)
```

Output:

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236614602-cf61cce1-a5c5-4e7f-92b9-dcd4aa03921f.png" width= 20%>
  <img src = "https://user-images.githubusercontent.com/51687528/236614613-226412e3-7040-41d0-8143-ca666d1b3cdb.png" width = 20%>
  <img src = "https://user-images.githubusercontent.com/51687528/236614625-43ad2800-5cb6-4172-bb6b-6a5823c95d34.png" width = 20%>
  <img src = "https://user-images.githubusercontent.com/51687528/236614635-a9f64dc4-7350-4df4-b996-2c6f075973a7.png" width = 20%>
</div>
<p align="center">
  <em>Examples of data augmentation</em>
</p>

-----
## Training the model
Once we have our final dataset, it is time to train the model. We will be using a pre-trained VGG16 architecture as it is a model that has consistently performed well.

### Function
`preparing_dataset(filepath, train_val_split, seed)`: Function that takes in the dataset filepath, the ratio split between test and val dataset, the seed for randomisation and creates the training and validation dataset. train_val_split is default at 0.2 and seed is default at 123.

`visualise_dataset(train_ds)`: Function that takes in the training dataset and shows the first 9 images of the first batch, help visualises data

`train_model(class_names, train_ds, val_ds, batch_size, epochs, modelpath)`: Function that takes in the class_names to create the model, training and validation dataset and the specifics of the model training. Default batch_size and epochs is 32 and 15 respectively. The model will be trained on the dataset and best model (based on lowest loss) will be saved to the filepath.

```
# How to use code
from Training_model import *

class_names is default as ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp']
datasetfilepath = "./Dataset"
modelpath = "./Models/best_model.h5"

train_val_split is default as 0.2
seed is default as 123
batch_size is default as 32
epoch is default as 15

class_names, train_ds, val_ds = preparing_dataset(datasetfilepath, train_val_split, seed)
visualise_dataset(train_ds)
train_model(class_names, train_ds, val_ds, batch_size, epochs, modelpath)
```

-----
## Running the model and linking to Firebase

### Function
`Hand_gesture_recognition(secretspath, url, class_names, modelpath):`: Function that takes in the file path to secrets.json file of firebase database, url of database, class names and the path to the best model. Function will constantly send live footage of the webcam to model. If a fist is detected, update the Firebase data to 0. If a high five is detected, update the Firebase data to 1.

```
#How to use code
from Gesture_Recognition import *

secretspath = "./Database/secrets.json"
url = "https://xxx.firebaseio.com"
class_names is default as ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp']
modelpath = "./Models/best_model.h5"

Hand_gesture_recognition(secretspath, url, class_names, modelpath)
```

Output:

<div float="left" align="center">
  <img src = "https://user-images.githubusercontent.com/51687528/236615631-791ea5d7-469b-4e00-84b3-359edc282f64.png" width= 30%>
  <img src = "https://user-images.githubusercontent.com/51687528/236615637-31fe3fb3-103b-4bb6-b70f-81080f75b74b.png" width = 50%>
</div>

-----
## Arduino
Create your Arudino Circuit based on the circuit diagram below and run the arduino code. 

<div align = "center">
  <img src="https://user-images.githubusercontent.com/51687528/236615576-db8a6e9a-d2b0-4ea8-b5e2-68d285e39c50.png" width = 70%>
</div>

The arduino code will constantly read the data in Firebase and on the LED if a 1 is detected or off the LED if a 0 is detected.






