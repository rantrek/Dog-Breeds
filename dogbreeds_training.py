import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import RMSprop, Adam

"""Training with CNN model"""

#Functions
def getModel(pretrained, input_shape):
    
    pretrained_model = pretrained(include_top=False, input_shape=input_shape,weights='imagenet', pooling= "max", classes=120)
    #Make model not trainable for use in feature extraction
    for layer in pretrained_model.layers:
        layer.trainable = False
    #Get last layer
    last_output = pretrained_model.layers[-1].output
    # Flatten the output layer to 1 dimension
    #res = Sequential()
    #res.add(pretrained_model)
    #res.add(Flatten())
    res = Flatten()(last_output)
    # Add a fully connected layer with 128 hidden units and ReLU activation
    res = Dense(128,activation="relu")(res)
    res = BatchNormalization()(res)
    #res.add(Dense(128,activation="relu"))
    res = Dense(128,activation="relu")(res)
    res = Dropout(0.2)(res)
    res = BatchNormalization()(res)
    #res.add(Dropout(0.2)) 
    #res.add(BatchNormalization()) 
    #res.add(Dense(120,activation="softmax"))
    res = Dense(120,activation="softmax")(res)
    
    # Configure and compile the model
    model = Model(pretrained_model.input, res)
    #model = res
    model.compile(loss='categorical_crossentropy',
                optimizer="Adam",
                metrics=['AUC', 'acc'])
    model.summary()
    return model

#Load train data and labels

X = np.load('train1.npy')
Y = np.load('labels.npy')
#test = np.load('test.npy')

#Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    stratify=Y, random_state=42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#Convert data type 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True
                                    )


valid_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(X_train)
valid_datagen.fit(X_test)
#test_datagen.fit(test)

training_set =  train_datagen.flow(X_train,Y_train,batch_size=20)
validation_set  =  valid_datagen.flow(X_test,Y_test,batch_size=20)

#Train the model
input_shape = (224,224,3)
model = getModel(MobileNetV2, input_shape)
#Fit the model
history = model.fit(training_set,epochs = 30, validation_data = validation_set, verbose =1)
model.evaluate(training_set,verbose=1)
model.evaluate(validation_set,verbose=1)
model.save('mobilenetv2_model.h5')

