import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))

train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"
additional_test_path = "data/additional_test"

image_shape = (350,350,3)
N_CLASSES = 2
BATCH_SIZE = 256

# defining the coefficient that our regularizer will use
weight_decay = 1e-3

# building a sequential CNN model and adding layers to it
# dropout and the regularizer are used in general to prevent overfitting
first_model = Sequential([
    Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 32 , kernel_size = 2, activation = 'relu',
           kernel_regularizer = regularizers.l2(weight_decay)),
    MaxPooling2D(pool_size = 2),
    
    Dropout(0.3),
    Flatten(),
    Dense(300,activation='relu'),
    Dropout(0.3),
    Dense(2,activation='softmax')
])

def train_validation_plot(history):
    # add history of accuracy and validation accuracy to the plot
    plt.plot(history.history['acc'], label = 'train',)
    plt.plot(history.history['val_acc'], label = 'valid')

    # adding legend and labels
    
    plt.title("Best Result for IgniteNN")
    plt.legend(loc = 'lower right')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.yticks(np.arange(0.5 ,1.0, 0.01)) 

    # show the plot
    plt.show()


def train_eval(train_set, validation_set, test_set):
    
    # saves the model so we can evaluate on the additional test set later
    checkpointer = ModelCheckpoint('first_model.hdf5',verbose=1, save_best_only= True)

    # early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)

    # From experimentation we found adam to be the best
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.00001, decay= 1e-5)

    first_model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics=['AUC','acc'])
    # first_model = load_model('first_model.hdf5') # comment this if running the first time
    first_model.summary()
    
    history = first_model.fit(train_set,
                    epochs = 50,
                    verbose = 1,
                    validation_data = validation_set,
                    callbacks = [checkpointer, early_stopping])
    
    train_validation_plot(history)
    evaluate("first_model.hdf5", test_set)

def evaluate(file, test_set):
    model = load_model(file)
    result = model.evaluate(test_set)

def main():

    train_gen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
    train_images = train_gen.flow_from_directory(train_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')


    val_gen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
    val_images = val_gen.flow_from_directory(valid_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')

 
    test_gen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
    test_images = test_gen.flow_from_directory(test_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')
    
    additional_gen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
    additional_images = additional_gen.flow_from_directory(additional_test_path,batch_size = 25, target_size = (350,350), class_mode = 'categorical')

    train_eval(train_images, val_images, test_images)
    evaluate("first_model.hdf5", additional_images)


if __name__ == '__main__':
    main()
