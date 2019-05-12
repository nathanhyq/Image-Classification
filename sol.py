# coding: utf-8
import numpy as np

from skimage.feature import hog

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
## load data

# Since the structure of Fashion-MNIST data is the same as MNIST, use the function in tensorflow designed to load MNIST to load data.
data_path = 'data'
mnist = input_data.read_data_sets('data')
mnist_train = mnist.train.images
train_label = np.asarray(mnist.train.labels, dtype=np.int32)
mnist_test = mnist.test.images
test_label = np.asarray(mnist.test.labels, dtype=np.int32)


## step I. extract features from Fashion-MNIST, containing:
#   1. pixel expanding
#   2. pca
#   3. hog
## pixel expanding

# define a function to construct Dataset
def Dataset(train_data, train_label, test_data, test_label):
    # normalize the data
    ss = StandardScaler()
    ss.fit(train_data)
    train_data = ss.transform(train_data)
    test_data = ss.transform(test_data)
    return {'train_data':train_data, 'train_label':train_label, 'test_data':test_data, 'test_label':test_label}

# pixel expanding	
pixel_dataset = Dataset(mnist_train, train_label, mnist_test, test_label)

## pca
pca_all = PCA(n_components=784)
pca_all.fit(mnist_train)
var_ratio = np.cumsum(pca_all.explained_variance_ratio_)

# 0.8
index1 = sum((var_ratio < 0.8))  # num of component that explained variance sum up to 80%
pca = PCA(n_components=index1)
mnist_train_pca = pca.fit_transform(mnist_train)
mnist_test_pca = pca.transform(mnist_test)
pca_dataset1 = Dataset(mnist_train_pca, train_label, mnist_test_pca, test_label)

# 0.9
index2 = sum((var_ratio < 0.9))  # num of component that explained variance sum up to 90%
pca = PCA(n_components=index2)
mnist_train_pca = pca.fit_transform(mnist_train)
mnist_test_pca = pca.transform(mnist_test)
pca_dataset2 = Dataset(mnist_train_pca, train_label, mnist_test_pca, test_label)


# explained variance ratio plot; give out the select point
plt.figure(1)
plt.plot(range(784), var_ratio)
plt.plot([index1], [var_ratio[index1]], marker='o', markersize=5, color='red')
plt.text(index1, var_ratio[index1], '(%d, %f)'%(index1, var_ratio[index1]))
plt.plot([index2], [var_ratio[index2]], marker='o', markersize=5, color='red')
plt.text(index2, var_ratio[index2], '(%d, %f)'%(index2, var_ratio[index2]))
plt.title('Explained Variance Ratio')
plt.xlabel('num of component')
plt.ylabel('ratio')
plt.show()


## hog
mnist_train_hog = np.zeros((len(mnist_train), 81))
for i in range(len(mnist_train)):
    mnist_train_hog[i,:] = hog(mnist_train[i,:].reshape((28,28)),block_norm='L2-Hys')

mnist_test_hog = np.zeros((len(mnist_test), 81))
for i in range(len(mnist_test)):
    mnist_test_hog[i,:] = hog(mnist_test[i,:].reshape((28,28)),block_norm='L2-Hys')

hog_dataset = Dataset(mnist_train_hog, train_label, mnist_test_hog, test_label)


## Step II model training: 
#    1. train support vector machine, random forest on pixel, pca and hog dataset
#    2. train CNN on raw images
#    3. test models on testing date and make comparison

# define a function to train SVM using Dataset defined, return testing accuracy
def svm_train(dataset):
    train_data, train_label = dataset['train_data'], dataset['train_label']
    test_data, test_label = dataset['test_data'], dataset['test_label']
    svm = SVC()
    svm.fit(train_data, train_label)
    return svm.score(test_data, test_label)

# define a function to train Random Forest using Dataset defined, return testing accuracy
def rf_train(dataset):
    train_data, train_label = dataset['train_data'], dataset['train_label']
    test_data, test_label = dataset['test_data'], dataset['test_label']
    rf = RandomForestClassifier()
    rf.fit(train_data, train_label)
    return rf.score(test_data, test_label)


# train and test svm on pixel, pca and hog dataset
acc_svm = []
acc_svm.append(svm_train(pixel_dataset))
acc_svm.append(svm_train(pca_dataset1))
acc_svm.append(svm_train(pca_dataset2))
acc_svm.append(svm_train(hog_dataset))
print(acc_svm)


# train and test random forest on pixel, pca and hog dataset
acc_rf = []
acc_rf.append(rf_train(pixel_dataset))
acc_rf.append(rf_train(pca_dataset1))
acc_rf.append(rf_train(pca_dataset2))
acc_rf.append(rf_train(hog_dataset))
print(acc_rf)


# train CNN on raw images
batch_size = 256
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

mnist_train_cnn = mnist_train.reshape(mnist_train.shape[0], img_rows, img_cols, 1)
mnist_test_cnn = mnist_test.reshape(mnist_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

mnist_train_cnn = mnist_train_cnn.astype('float32')
mnist_test_cnn = mnist_test_cnn.astype('float32')
mnist_train_cnn /= 255
mnist_test_cnn /= 255

# convert class vectors to binary class matrices
train_label_cnn = keras.utils.to_categorical(train_label, num_classes)
test_label_cnn = keras.utils.to_categorical(test_label, num_classes)

# divide training data into training and validation set
train_data, vali_data, train_label_cnn2, vali_label_cnn = train_test_split(mnist_train_cnn, train_label_cnn, 
                                                                           test_size=0.2, random_state=200)
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, train_label_cnn2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(vali_data, vali_label_cnn))
score = model.evaluate(mnist_test_cnn, test_label_cnn, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# view the structure of CNN
model.summary()

## visualize the loss and accuracy of training
hist = history.history
loss_train, acc_train, loss_val, acc_val = hist['loss'], hist['acc'], hist['val_loss'], hist['val_acc']


n = len(loss_train)
plt.plot(range(n),loss_train,label='train',color='red')
plt.plot(range(n),loss_val,label='val',color='blue')
plt.title('Loss of Training and Validation Set')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.plot(range(n),acc_train,label='train',color='red')
plt.plot(range(n),acc_val,label='val',color='blue')
plt.title('Accuracy of Training and Validation Set')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()


## Step III model training:
#    1. extract features using the hidden layer of CNN
#    2. re-train support vector machine and random forest using new features and make comparison

# define a function to get value of hidden layer output
get_hidden_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                     [model.layers[10].output])

# output in test mode = 0
import numpy as np
train_data_cnn = np.zeros((55000,128))
for i in range(5500):
    train_data_cnn[i*10:i*10+10,] = get_hidden_layer_output([mnist_train_cnn[i*10:i*10+10,], 0])[0]
test_data_cnn = get_hidden_layer_output([mnist_test_cnn,0])[0]

cnn_dataset = Dataset(train_data_cnn, train_label, test_data_cnn, test_label)

print(svm_train(cnn_dataset))

print(rf_train(cnn_dataset))








