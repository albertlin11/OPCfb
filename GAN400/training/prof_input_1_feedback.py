import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def create_mainflow(input_size, model_name):
    inputs = Input(input_size)
    
    x1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    x1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    x1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    x1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    x1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    output = Conv2D(1, 1, activation='sigmoid')(x1)
    
    mainflow = Model(inputs=inputs, outputs=output, name=model_name)
    
    return mainflow

def prof_input_1_feedback(args, pretrained_weights = None,input_size = (256,256,1)):
    
    mainflow = create_mainflow(input_size, "mainflow")
    
    Mask1 = Input(input_size)

    SEM1=mainflow(Mask1)
    SEM2=mainflow(SEM1)
    model = Model(inputs = Mask1, outputs = SEM2, name='feedback')

    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


