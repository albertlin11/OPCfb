import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def model_base(args, pretrained_weights = None,input_size = (256,256,1)):
    input_size = (args.image_size, args.image_size, 1)
    inputs = Input(input_size)
    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    output = Conv2D(1, 1, activation = 'sigmoid')(x1)

    model = Model(inputs = inputs, outputs = output)
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


