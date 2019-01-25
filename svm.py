import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
import tensorflow as tf
#from sklearn import SVM
import tensorboard
from time import time
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier


#min_max_scaler = preprocessing.normalize()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

path="/home/shakil/Desktop/violent_flow_detection/Data/"

labels=['violent','non_violent']
'''
for label in labels:
    vector=[]

    images = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
    for image in images:
        img=cv2.imread(image,0)
        vector.append(img)
    np.save(label + '.npy', vector)

'''

X_train = np.load('train_videos/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('train_videos/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))

assert X_train.shape[0] == len(y_train)


X_test = np.load('test_videos/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('test_videos/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))

assert X_test.shape[0] == len(y_test)

#x_train, x_test,y_train,y_test= train_test_split(X, y, test_size= .2, random_state=42, shuffle=True)

y_train_hot=to_categorical(y_train)
y_test_hot=to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


model=Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.l2(0.01), input_shape=(28, 28, 1)))
model.add(Dropout(0.50))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.50))
model.add(BatchNormalization())
#model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.50))
#model.add(Dense(10,activation='relu'))
model.add(Dense(2, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
#model.load_weights('movie_data.h5')
'''
for layer in model.layers[-7:]:
    layer.trainable = False
'''

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train_hot, batch_size=100, epochs=1, verbose=1,validation_data=(X_test,y_test_hot), callbacks=[keras.callbacks.TensorBoard(log_dir="logs/TRANSFER/50epochs{}".format(time()), histogram_freq=0, write_graph=False, write_images=False)]
)
score,acc=model.evaluate(X_test, y_test_hot, batch_size=100)

pred=model.predict(X_test)
pred=np.argmax(pred,axis=1)
y_test_hot=np.argmax(y_test_hot,axis=1)
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_hot,pred))
#print(confusion_matrix(y_test_hot,pred))

print (acc)
print(y_train[:100])
print(y_train.shape)
model_feat = Model(inputs=model.input,outputs=model.layers[1].output)
feat_train = model_feat.predict(X_train)
feat_test = model_feat.predict(X_test)

length=len(feat_train)
feat_train=feat_train.reshape(length,-1)

length=len(feat_test)
feat_test=feat_test.reshape(length,-1)

K_NN=KNeighborsClassifier()
K_NN.fit(feat_train,y_train.ravel())
accuracy_knn=K_NN.score(feat_test,y_test)
print(accuracy_knn)

svm = SVC(kernel='rbf')
svm.fit(feat_train,y_train[:100])
acc=svm.score(X_test,y_test[1:5])
print(acc)



    



