#PERCEPTRON MODEL
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import Perceptron
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('mnist_train.csv')
df_test=pd.read_csv('mnist_test.csv')
#print(df.head())
#print(df.isnull().sum())
#print(df.isnull().sum().sum())
#print(df.info())
#print(df.shape)
#print(df_test.shape)
x=df.drop('label',axis=1).values
y=df['label'].values
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
x_test=df_test.drop('label',axis=1).values
y_test=df_test['label'].values

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0
x_val=x_val.astype("float32")/255.0 

y_train_cat=tf.keras.utils.to_categorical(y_train,10)
y_test_cat=tf.keras.utils.to_categorical(y_test,10)
y_val_cat=tf.keras.utils.to_categorical(y_val,10)

x_train_img=x_train.reshape(-1,28,28)
x_test_img=x_test.reshape(-1,28,28)
x_val_img=x_val.reshape(-1,28,28)

perceptron=keras.Sequential([
    Flatten(input_shape=(28,28)),
    layers.Dense(10,activation='softmax')
])

perceptron.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

history_percep=perceptron.fit(x_train_img,y_train_cat,
                       validation_data=(x_val_img,y_val_cat),
                       epochs=5,batch_size=32,verbose=0)





ann=keras.Sequential([
    layers.Input(x_train_img.shape[1:]),
    Flatten(),
    layers.Dense(128,activation='relu'),
    Dropout(0.3),
    layers.Dense(64,activation='relu'),
    Dropout(0.3),
    layers.Dense(10,activation='softmax')
])

ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history_ann=ann.fit(x_train_img,y_train_cat,
                validation_data=(x_val_img,y_val_cat),
                epochs=5,batch_size=32,verbose=0)




x_train_cnn=x_train.reshape(-1,28,28,1)
x_val_cnn=x_val.reshape(-1,28,28,1)
x_test_cnn=x_test.reshape(-1,28,28,1)

cnn=keras.Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    layers.Dense(128,activation='relu'),
    Dropout(0.5),
    layers.Dense(10,activation='softmax')

])

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history_cnn=cnn.fit(x_train_cnn,y_train_cat,
                validation_data=(x_val_cnn,y_val_cat),
                epochs=5,batch_size=32,verbose=0)

def plot_training(history, title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.show()

plot_training(history_percep, "Perceptron")
plot_training(history_ann, "ANN")
plot_training(history_cnn, "CNN")

plt.figure(figsize=(10,6))
plt.plot(history_percep.history['val_accuracy'], label="Perceptron")
plt.plot(history_ann.history['val_accuracy'], label="ANN")
plt.plot(history_cnn.history['val_accuracy'], label="CNN")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Val Accuracy")
plt.legend()
plt.show()     

def show_side_by_side(models, model_names, X, X_cnn, y_true, n=5):
    print("Function started")
    idxs = np.random.choice(len(X), n, replace=False)

    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(idxs):

        # Predictions
        percep_pred = np.argmax(models[0].predict(X[idx].reshape(1, 28, 28), verbose=0))
        ann_pred = np.argmax(models[1].predict(X[idx].reshape(1, 28, 28), verbose=0))
        cnn_pred = np.argmax(models[2].predict(X_cnn[idx].reshape(1, 28, 28, 1), verbose=0))

        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[idx], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"True: {y_true[idx]}\n"
            f"P: {percep_pred}\n"
            f"ANN: {ann_pred}\n"
            f"CNN: {cnn_pred}",
            fontsize=9
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


# Call it OUTSIDE the function, at the top level
show_side_by_side(
    [perceptron, ann, cnn],
    ["Perceptron", "ANN", "CNN"],
    x_test_img,
    x_test_cnn,
    y_test,
    5
)