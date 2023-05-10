import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Activation, AveragePooling2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt

for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))

train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"
additional_test_path = 'data/additional_test'

# implemented the identity and convolutional blocks according to the original ResNet diagram
def res_identity(x, filters): 

  x_skip = x
  f1, f2 = filters

  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x


def res_conv(x, s, filters):
  
  x_skip = x
  f1, f2 = filters

  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x


def resIgnite():

  input_img = tf.keras.Input(shape=(350,350,3))
  x = ZeroPadding2D(padding=(3, 3))(input_img)

  x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = res_conv(x, s=1, filters=(16, 32))
  x = res_identity(x, filters=(16, 32))

  x = res_conv(x, s=2, filters=(32, 64))
  x = res_identity(x, filters=(32, 64))
  x = res_identity(x, filters=(32, 64))

  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  x = Dense(2, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  # defines the model 
  model = tf.keras.Model(inputs=input_img, outputs=x, name='resIgnite')

  return model

def train_validation_plot(history):
    
    plt.plot(history.history['acc'], label = 'train',)
    plt.plot(history.history['val_acc'], label = 'valid')

    plt.title("Best Result for resIgnite")
    plt.legend(loc = 'lower right')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')                                                                                                                                                                                                                  
    plt.yticks(np.arange(0.5 ,1.0, 0.01)) 

    # show the plot
    plt.show()


def train_eval(train_set, validation_set, test_set):

    checkpointer2 = ModelCheckpoint('second_model.hdf5',verbose=1, save_best_only= True)

    # early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)

    # From experimentation we found adam to be the best
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.00001, decay= 1e-5)

    second_model = resIgnite()
    # second_model = load_model('second_model.hdf5') # comment this if running the first time
    second_model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics=['AUC','acc'])
    
    second_model.summary()

    history = second_model.fit(train_set,
                    epochs = 50,
                    verbose = 1,
                    validation_data = validation_set,
                    callbacks = [checkpointer2, early_stopping])
    
    train_validation_plot(history)
    evaluate('second_model.hdf5', test_set)

def evaluate(file, test_set):
    model = load_model(file)
    result = model.evaluate(test_set)


def main():

    # Uses ImageDataGenerator to transform our path into dataset object that keras can extract from in training.
    train_gen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
    train_images = train_gen.flow_from_directory(train_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')


    val_gen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
    val_images = val_gen.flow_from_directory(valid_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')

 
    test_gen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
    test_images = test_gen.flow_from_directory(test_path, batch_size = 256, target_size = (350,350), class_mode = 'categorical')
    
    additional_gen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
    additional_images = additional_gen.flow_from_directory(additional_test_path,batch_size = 25, target_size = (350,350), class_mode = 'categorical')

    train_eval(train_images, val_images, test_images) # trains the model
    evaluate('second_model.hdf5', additional_images) # evaluates our additional test set
    

if __name__ == '__main__':
    main()
