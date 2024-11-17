import numpy as np 
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class CausalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.Attention(**kwargs)
    self.add = tf.keras.layers.Add() 
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    attn = self.mha([x,x]) # x is the query and key and value
    residue = Add()([x, attn]) # in self attention we add SEM1 as residual connection
    return self.layernorm(residue)
  
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

def prof_input_1_feedback_self_a(args, pretrained_weights = None,input_size = (256,256,1)):
    input_size = (args.image_size, args.image_size, 1)
    mainflow = create_mainflow(input_size, "mainflow")
    
    
    Mask1 = Input(input_size)
    
    SEM1 = mainflow(Mask1)
    SEM1_reshape = Reshape((args.image_size,args.image_size))(SEM1)
    self_attention = CausalSelfAttention(dropout=0.1)
    Mask2 = self_attention(SEM1_reshape)
    Mask2_reshape = Reshape((args.image_size,args.image_size,1))(Mask2)


    SEM2=mainflow(Mask2_reshape)
    model = Model(inputs = Mask1, outputs = SEM2, name='feedback_self')


    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


