#################################################################
########################## IMPORTS ##############################
#################################################################

import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory

"""
This file is designed to implement CNN for image classification. 
Our application focus on classifying whether an image contains one of the 12 wonders listed in dataset.
"""

#################################################################
###################### BASE PARAMETERS ##########################
#################################################################

CP_F = 0
learning_rate = 0.001
epochs = 20
init_threshold = 0.5

path = 'C:/Users/duboisrouvray/Documents/Entrainement_ML/WondersofWorld/'

n_classes = 12
image_height = 200
image_width = image_height
n_channel = 3
batch_size = 2

#################################################################
########################## DATASET ##############################
#################################################################

data_dir_train = pathlib.Path('C:/Users/duboisrouvray/Documents/Entrainement_ML/WondersofWorld')

train_dataset = image_dataset_from_directory(
    directory = data_dir_train,
    subset = "training",
    shuffle = True,
    validation_split = 0.2,
    seed=42,
    batch_size = batch_size,
    image_size = (image_height, image_width)
)

validation_dataset = image_dataset_from_directory(
    directory = data_dir_train,
    subset = "validation",
    shuffle = True,
    validation_split = 0.2,
    seed=42,
    batch_size = batch_size,
    image_size = (image_height, image_width)
)


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#################################################################
########################### MODEL ###############################
#################################################################

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

rescale = tf.keras.layers.Rescaling(1./127.5, offset = -1)
inputs = tf.keras.Input(shape=(200, 200, 3))
model = tf.keras.Sequential([
    tf.keras.Input(shape=(200, 200, 3)),
    data_augmentation,
    rescale,
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(12, activation = 'softmax'),
    ])
model.summary()

#################################################################
######################### CALLBACKS #############################
#################################################################

if (CP_F) :
    checkpoint_path = "C:/Users/duboisrouvray/Documents/Entrainement_ML/checkpoints_Wonders/weights.04-0.6463.cpkt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)
    
earlystop_callback = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="C:/Users/duboisrouvray/Documents/Entrainement_ML/checkpoints_Wonders/weights.{epoch:02d}-{val_accuracy:.4f}.cpkt",
                                                 save_weights_only=True,
                                                 monitor='val_accuracy',
                                                 save_best_only=True,
                                                 mode='max',
                                                 initial_value_threshold = init_threshold,
                                                 verbose=1)

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

plt_callback = PlotLearning()

#################################################################
########################## TRAINING #############################
#################################################################

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics = ['accuracy'])

history = model.fit(train_dataset,
                    epochs= epochs,
                    validation_data = validation_dataset,
                    callbacks=[cp_callback, earlystop_callback])

learning_rate = learning_rate/10

history = model.fit(train_dataset,
                    epochs= epochs,
                    validation_data = validation_dataset,
                    callbacks=[cp_callback, earlystop_callback])

learning_rate = learning_rate/2

history = model.fit(train_dataset,
                    epochs= epochs,
                    validation_data = validation_dataset,
                    callbacks=[cp_callback, earlystop_callback])

#################################################################
################### TRAINING VISUALIZATION ######################
#################################################################

# summarize history for accuracy
plt.style.use('dark_background')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
