#Importing libraries and packages

#Data management, analysis and visualization
import numpy as np
import pandas as pd
#from keras.preprocessing import image
import keras.utils as image
from sklearn.preprocessing import LabelEncoder

"""Data processing"""


#Functions

def addColumn (data,f):
    #A function that adds a third column 'path' to the labels dataframes
    #Appending the path to the image in the path column.
    data = data.assign(path=lambda x: f + x['id'] +'.jpg')
    return data

#function to convert image files to numpy array

def convert_img(path):
    #path - path to the image file
    #returns image as numpy array
    img = image.load_img(path, target_size = (224,224))
    img = image.img_to_array(img)
    return img

#Loading paths to the train and test images

train= 'dog-breed-identification/train/'
test = 'dog-breed-identification/test/'

#Load train and test labels csv files
labels = pd.read_csv('labels.csv')
testf = pd.read_csv('sample_submission.csv')

#Get the number of classes
labelsb = np.unique(labels.breed)
numClasses = labelsb.size
print(numClasses)

#Add column for image path to the train and test labels dataframes
labels = addColumn(labels,train)
test_data = addColumn(testf,test)

#Convert train and test images into numpy arrays and resize the images
X =np.array([convert_img(img)
                    for img in labels['path'].values.tolist()])
X.shape

test_img = np.array([convert_img(img)
                   for img in test_data['path'].values.tolist()])
test_img.shape

#Applying  label encoder to the breeds column in labels breed column
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(labels['breed'].values)
#Convert 1-dimensional class arrays to 30-dimensional class matrices for labels[breed]
Y= image.np_utils.to_categorical(Y, 120)
Y.shape

#Save train and test data as npy files

np.save('train1.npy',X)
#np.save('labels.npy',Y)
np.save('test1.npy',test_img)



