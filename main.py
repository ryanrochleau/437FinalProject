import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.metrics import Precision,Recall,AUC
from keras.utils import np_utils
from keras.layers import Dropout,Dense,Flatten,MaxPool2D,Conv2D
from sklearn.model_selection import train_test_split
from PIL import Image

# Hyperparameters for the Keras.fit method
# The dataset contains 43 classes so changing the
# value below will train the models on more
# classes of data

classCount = 43
epochs = 10
batchSize = 128

# Loads the data from the traffic data set
def loadData():
    cleanedData = []
    labels = []
    # The labels for each class are simply values from 0 to 42
    for label in range(classCount):
        # Generate a string for the directories containing the training data
        classDirectory = os.getcwd() + '\\train\\' + str(label)
        trainingData = os.listdir(classDirectory)
        print(label)
        for sample in trainingData:
            # Open the image for resizing and to append
            data = Image.open(classDirectory + '\\' + sample)
            data = data.resize((32,32))
            data = np.array(data)

            # Append the data alongside its label
            cleanedData.append(data)
            labels.append(label)
    return np.array(cleanedData), np.array(labels)


def createCNNModels(inputShape):
    models = []
    # Relu CNN
    CNN = keras.models.Sequential()
    CNN.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu',input_shape=inputShape))
    CNN.add(Dropout(rate=(0.5)))
    CNN.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu'))
    CNN.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN.add(MaxPool2D(pool_size=(2,2)))
    CNN.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN.add(Flatten())
    CNN.add(Dense(classCount,activation='softmax'))
    CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Relu', CNN))

    # Tanh CNN
    CNN2 = keras.models.Sequential()
    CNN2.add(Conv2D(filters=64,kernel_size=(6,6),activation='tanh',input_shape=inputShape))
    CNN2.add(Dropout(rate=(0.5)))
    CNN2.add(Conv2D(filters=64,kernel_size=(6,6),activation='tanh'))
    CNN2.add(Conv2D(filters=64, kernel_size=(6,6), activation='tanh'))
    CNN2.add(MaxPool2D(pool_size=(2,2)))
    CNN2.add(Conv2D(filters=128, kernel_size=(6, 6), activation='tanh'))
    CNN2.add(Flatten())
    CNN2.add(Dense(classCount,activation='softmax'))
    CNN2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Tanh',CNN2))

    # Sigmoid
    CNN10 = keras.models.Sequential()
    CNN10.add(Conv2D(filters=64,kernel_size=(6,6),activation='sigmoid',input_shape=inputShape))
    CNN10.add(Dropout(rate=(0.5)))
    CNN10.add(Conv2D(filters=64,kernel_size=(6,6),activation='sigmoid'))
    CNN10.add(Conv2D(filters=64, kernel_size=(6,6), activation='sigmoid'))
    CNN10.add(MaxPool2D(pool_size=(2,2)))
    CNN10.add(Conv2D(filters=128, kernel_size=(6, 6), activation='sigmoid'))
    CNN10.add(Flatten())
    CNN10.add(Dense(classCount,activation='softmax'))
    CNN10.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('sigmoid',CNN10))

    # Dropout 0.75
    CNN3 = keras.models.Sequential()
    CNN3.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu',input_shape=inputShape))
    CNN3.add(Dropout(rate=(0.75)))
    CNN3.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu'))
    CNN3.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN3.add(MaxPool2D(pool_size=(2,2)))
    CNN3.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN3.add(Flatten())
    CNN3.add(Dense(classCount,activation='softmax'))
    CNN3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Dropout 0.75', CNN3))

    # Dropout 0.5
    CNN4 = keras.models.Sequential()
    CNN4.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu',input_shape=inputShape))
    CNN4.add(Dropout(rate=(0.5)))
    CNN4.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu'))
    CNN4.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN4.add(MaxPool2D(pool_size=(2,2)))
    CNN4.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN4.add(Flatten())
    CNN4.add(Dense(classCount,activation='softmax'))
    CNN4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Dropout 0.5', CNN4))

    # Dropout 0.25
    CNN5 = keras.models.Sequential()
    CNN5.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu',input_shape=inputShape))
    CNN5.add(Dropout(rate=(0.25)))
    CNN5.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu'))
    CNN5.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN5.add(MaxPool2D(pool_size=(2,2)))
    CNN5.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN5.add(Flatten())
    CNN5.add(Dense(classCount,activation='softmax'))
    CNN5.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Dropout 0.25', CNN5))

    # No Dropout
    CNN6 = keras.models.Sequential()
    CNN6.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu',input_shape=inputShape))
    CNN6.add(Dropout(rate=(0)))
    CNN6.add(Conv2D(filters=64,kernel_size=(6,6),activation='relu'))
    CNN6.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN6.add(MaxPool2D(pool_size=(2,2)))
    CNN6.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN6.add(Flatten())
    CNN6.add(Dense(classCount,activation='softmax'))
    CNN6.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[Recall(),Precision(),'accuracy',AUC()])
    models.append(('Dropout 0', CNN6))

    # 2x2 Kernal
    CNN7 = keras.models.Sequential()
    CNN7.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', input_shape=inputShape))
    CNN7.add(Dropout(rate=(0.5)))
    CNN7.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    CNN7.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    CNN7.add(MaxPool2D(pool_size=(2, 2)))
    CNN7.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu'))
    CNN7.add(Flatten())
    CNN7.add(Dense(classCount, activation='softmax'))
    CNN7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(), Precision(), 'accuracy', AUC()])
    models.append(('2x2 Kernel', CNN7))

    # 4x4 Kernal
    CNN8 = keras.models.Sequential()
    CNN8.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu', input_shape=inputShape))
    CNN8.add(Dropout(rate=(0.5)))
    CNN8.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    CNN8.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    CNN8.add(MaxPool2D(pool_size=(2, 2)))
    CNN8.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
    CNN8.add(Flatten())
    CNN8.add(Dense(classCount, activation='softmax'))
    CNN8.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(), Precision(), 'accuracy', AUC()])
    models.append(('4x4 Kernal', CNN8))

    # 6x6 Kernal
    CNN9 = keras.models.Sequential()
    CNN9.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu', input_shape=inputShape))
    CNN9.add(Dropout(rate=(0.5)))
    CNN9.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN9.add(Conv2D(filters=64, kernel_size=(6, 6), activation='relu'))
    CNN9.add(MaxPool2D(pool_size=(2, 2)))
    CNN9.add(Conv2D(filters=128, kernel_size=(6, 6), activation='relu'))
    CNN9.add(Flatten())
    CNN9.add(Dense(classCount, activation='softmax'))
    CNN9.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(), Precision(), 'accuracy', AUC()])
    models.append(('6x6 Kernal', CNN9))

    return models

def plotAccuracyAndROC(fitResults):
    # Plot Accuracy for activation functions
    plt.figure(0)
    plt.plot(fitResults[0].history['val_accuracy'],label='ReLU Test Accuracy')
    plt.plot(fitResults[1].history['val_accuracy'], label='Tanh Test Accuracy')
    plt.plot(fitResults[2].history['val_accuracy'], label='Sigmoid Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('plot' + str(0) + '.png')

    # Plot AUC for activation functions
    plt.figure(1)
    plt.plot(fitResults[0].history['val_auc'],label='ReLU Test AUC')
    plt.plot(fitResults[1].history['val_auc_1'], label='Tanh Test AUC')
    plt.plot(fitResults[2].history['val_auc_2'], label='Sigmoid Test AUC')
    plt.title('Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Test AUC')
    plt.legend()
    plt.savefig('plot' + str(1) + '.png')

    # Plot Precision for activation functions
    plt.figure(6)
    plt.plot(fitResults[0].history['val_precision'],label='ReLU Test Precision')
    plt.plot(fitResults[1].history['val_precision_1'], label='Tanh Test Precision')
    plt.plot(fitResults[2].history['val_precision_2'], label='Sigmoid Test Precision')
    plt.title('Test Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Test Precision')
    plt.legend()
    plt.savefig('plot' + str(6) + '.png')

    # Plot Recall for activation functions
    plt.figure(9)
    plt.plot(fitResults[0].history['val_recall'],label='ReLU Test Recall')
    plt.plot(fitResults[1].history['val_recall_1'], label='Tanh Test Recall')
    plt.plot(fitResults[2].history['val_recall_2'], label='Sigmoid Test Recall')
    plt.title('Test Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Test Recall')
    plt.legend()
    plt.savefig('plot' + str(9) + '.png')

    # Plot Accuracy for Dropout values
    plt.figure(2)
    plt.plot(fitResults[3].history['val_accuracy'],label='0.75 Dropout Test Accuracy')
    plt.plot(fitResults[4].history['val_accuracy'], label='0.5 Dropout Test Accuracy')
    plt.plot(fitResults[5].history['val_accuracy'], label='0.25 Dropout Test Accuracy')
    plt.plot(fitResults[6].history['val_accuracy'], label='0 Dropout Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('plot' + str(2) + '.png')

    # Plot AUC for Dropout values
    plt.figure(3)
    plt.plot(fitResults[3].history['val_auc_3'],label='0.75 Dropout Test AUC')
    plt.plot(fitResults[4].history['val_auc_4'], label='0.5 Dropout Test AUC')
    plt.plot(fitResults[5].history['val_auc_5'], label='0.25 Dropout Test AUC')
    plt.plot(fitResults[6].history['val_auc_6'], label='0 Dropout Test AUC')
    plt.title('Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Test AUC')
    plt.legend()
    plt.savefig('plot' + str(3) + '.png')

    # Plot Precision for Dropout values
    plt.figure(7)
    plt.plot(fitResults[3].history['val_precision_3'],label='0.75 Dropout Test Precision')
    plt.plot(fitResults[4].history['val_precision_4'], label='0.5 Dropout Test Precision')
    plt.plot(fitResults[5].history['val_precision_5'], label='0.25 Dropout Test Precision')
    plt.plot(fitResults[6].history['val_precision_6'], label='0 Dropout Test Precision')
    plt.title('Test Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Test Precision')
    plt.legend()
    plt.savefig('plot' + str(7) + '.png')

    # Plot Recall for Dropout values
    plt.figure(10)
    plt.plot(fitResults[3].history['val_recall_3'],label='0.75 Dropout Test Recall')
    plt.plot(fitResults[4].history['val_recall_4'], label='0.5 Dropout Test Recall')
    plt.plot(fitResults[5].history['val_recall_5'], label='0.25 Dropout Test Recall')
    plt.plot(fitResults[6].history['val_recall_6'], label='0 Dropout Test Recall')
    plt.title('Test Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Test Recall')
    plt.legend()
    plt.savefig('plot' + str(10) + '.png')

    # Plot accuracy for Kernal values
    plt.figure(4)
    plt.plot(fitResults[7].history['val_accuracy'],label='2x2 Kernal Test Accuracy')
    plt.plot(fitResults[8].history['val_accuracy'], label='4x4 Kernal Test Accuracy')
    plt.plot(fitResults[9].history['val_accuracy'], label='6x6 Test Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('plot' + str(4) + '.png')

    # Plot AUC for Kernal values
    plt.figure(5)
    plt.plot(fitResults[7].history['val_auc_7'],label='2x2 Kernal Test AUC')
    plt.plot(fitResults[8].history['val_auc_8'], label='4x4 Kernal Test AUC')
    plt.plot(fitResults[9].history['val_auc_9'], label='6x6 Test AUC')
    plt.title('Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Test AUC')
    plt.legend()
    plt.savefig('plot' + str(5) + '.png')

    # Plot Precision for Kernal Values
    plt.figure(8)
    plt.plot(fitResults[7].history['val_precision_7'],label='2x2 Kernal Test Precision')
    plt.plot(fitResults[8].history['val_precision_8'], label='4x4 Kernal Test Precision')
    plt.plot(fitResults[9].history['val_precision_9'], label='6x6 Test Precision')
    plt.title('Test Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Test Precision')
    plt.legend()
    plt.savefig('plot' + str(8) + '.png')

    # Plot Recall for Kernal Values
    plt.figure(11)
    plt.plot(fitResults[7].history['val_recall_7'],label='2x2 Kernal Test Recall')
    plt.plot(fitResults[8].history['val_recall_8'], label='4x4 Kernal Test Recall')
    plt.plot(fitResults[9].history['val_recall_9'], label='6x6 Test Recall')
    plt.title('Test Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Test Recall')
    plt.legend()
    plt.savefig('plot' + str(11) + '.png')

# Defining the CNN
if __name__ == '__main__':
    # Loading the data
    X, y = loadData()

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

    # Constructing the models
    CNNModels = createCNNModels(X_train.shape[1:])

    # Convert the labels to one hot encoding matrices
    y_train = np_utils.to_categorical(y_train,classCount)
    y_test = np_utils.to_categorical(y_test,classCount)

    # Storing the History object return of the models
    fitResults = []

    # Fit each model
    for name,model in CNNModels:
        print(name)
        fitResults.append(model.fit(X_train,y_train,batch_size = batchSize,epochs = epochs,validation_data=(X_test,y_test)))

    plotAccuracyAndROC(fitResults)



