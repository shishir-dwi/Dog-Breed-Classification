#!/usr/bin/env python
# coding: utf-8

# In[3]:


# load all required libraries for Dog's Breed Identification Project
import cv2
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils.multiclass import unique_labels


# In[10]:


#read the csv file
df_labels = pd.read_csv("D:\\Dog Breed Classification\\Data Set\\labels.csv")
#store training and testing images folder location
train_file = "D:\\Dog Breed Classification\\Data Set\\train\\"
test_file = "D:\\Dog Breed Classification\\Data Set\\test\\"


# In[11]:


print(df_labels)


# In[12]:


print("Total number of unique Dog Breeds :",len(df_labels.breed.unique()))


# In[13]:


#specify number
num_breeds = 120
im_size = 224
batch_size = 64
encoder = LabelEncoder()


# In[14]:


#get all 120 unique breeds record 
breed_dict = list(df_labels['breed'].value_counts().keys()) 
new_list = sorted(breed_dict,reverse=True)[:num_breeds+1:1]
#change the dataset to have those 120 unique breed records
df_labels = df_labels.query('breed in @new_list')


# In[15]:


#create new column which will contain image name with the image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")
print(df_labels['img_file'])


# In[9]:


#create a numpy array of the shape
#(number of dataset records, image size , image size, 3 for rgb channel ayer)
#this will be input for model
train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')
 
#iterate over img_file column of our dataset
for i, img_id in enumerate(df_labels['img_file']):
  #read the image file and convert into numeric format
  #resize all images to one dimension i.e. 224x224
  #we will get array with the shape of
  # (224,224,3) where 3 is the RGB channels layers
  img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_COLOR),((im_size,im_size)))
  #scale array into the range of -1 to 1.
  #preprocess the array and expand its dimension on the axis 0 
  img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
  #update the train_x variable with new element
  train_x[i] = img_array
  


# In[10]:


#This will be the target for the model.
#convert breed names into numerical format
train_y = encoder.fit_transform(df_labels["breed"].values)
train_y


# In[11]:


#split the dataset in the ratio of 80:20. 
#80% for training and 20% for testing purpose
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)


# In[12]:


x_train


# In[13]:


#Image augmentation using ImageDataGenerator class
train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
 
#generate images for training sets 
train_generator = train_datagen.flow(x_train, 
                                     y_train, 
                                     batch_size=batch_size)
 
#same process for Testing sets also by declaring the instance
test_datagen = ImageDataGenerator()
 
test_generator = test_datagen.flow(x_test, 
                                     y_test, 
                                     batch_size=batch_size)


# In[4]:


#building the model using MobileNetV2 with input shape of our image array
#weights for our network will be from of imagenet dataset
#we will not include the first Dense layer
model =  MobileNetV2(weights='imagenet', include_top=False)

#falttening output & adding Fully-Connected Layers
x= model.output
x= BatchNormalization()(x)
x= GlobalAveragePooling2D()(x)
x= Dropout(0.5)(x)




# In[5]:


#add output layer having the shape equal to number of breeds
preds = Dense(num_breeds, activation='softmax')(x)
 
#create model class with inputs and outputs
model = Model(inputs=model.input, outputs=preds)

model.summary()


# In[17]:


for layers in model.layers[:-4]:
    layers.trainable = False
    
model.summary()


# In[39]:


#epochs for model training and learning rate for optimizer
epochs = 20
learning_rate = 1e-3
   
#using RMSprop optimizer to compile or build the model
optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
 
#fit the training generator data and train the model
hist = model.fit(train_generator,
                 steps_per_epoch= x_train.shape[0] // batch_size,
                 epochs= epochs,
                 validation_data= test_generator,
                 validation_steps= x_test.shape[0] // batch_size)
 
#Save the model for prediction
model.save("model")


# In[18]:


model.evaluate(x_test, y_test)


# In[19]:


def plot_sample(x, y, index):
    plt.figure(figsize = (30,4))
    plt.imshow(x[index])
    plt.xlabel(df_labels['breed'][y[index]])
    
plot_sample(x_test, y_test, 1)


# In[53]:


y_pred = model.predict(x_test)
y_pred


# In[54]:


y_result = [np.argmax(element) for element in y_pred]
y_result


# In[55]:


y_test


# In[20]:


plot_sample(x_test, y_test, 57)


# In[21]:


df_labels['breed'][y_result[57]]


# In[69]:


print("Classsification Report : \n\n", classification_report(y_test, y_result))


# In[85]:


def plot(x,y):
    labels = unique_labels(y_test)
    column = [f'Predicted{label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(x,y), columns = column, index = indices)
    
    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')


# In[88]:


plot(y_test, y_result)


# In[16]:


#load the model
model = load_model("model")
 
#get the image of the dog for prediction
pred_img_path = "C:\\Users\\asus\\Dropbox\\My PC (LAPTOP-KVI3FLV1)\\Downloads\\Sonu.jpg"
#read the image file and convert into numeric format
#resize all images to one dimension i.e. 224x224
pred_img_array = cv2.resize(cv2.imread(pred_img_path,cv2.IMREAD_COLOR),((im_size,im_size)))
#scale array into the range of -1 to 1.
#expand the dimension on the axis 0 and normalize the array values
pred_img_array = preprocess_input(np.expand_dims(np.array(pred_img_array[...,::-1].astype(np.float32)).copy(), axis=0))
 
#feed the model with the image array for prediction
pred_val = model.predict(np.array(pred_img_array,dtype="float32"))
 
#display the image of dog
#im= Image.open(pred_img_path)
#im.show()

plt.title("Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
 
image = mpimg.imread(pred_img_path)
plt.imshow(image)
plt.show()

#display the predicted breed of dog
pred_breed = sorted(new_list)[np.argmax(pred_val)]
print("Predicted Breed for this Dog is :",pred_breed)


# In[6]:





# In[ ]:




