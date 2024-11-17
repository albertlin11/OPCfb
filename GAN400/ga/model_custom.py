import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.w1 = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True, name = 'w1')
        self.b1 = tf.Variable(initial_value=b_init(shape=(input_dim, units), dtype="float32"), trainable=True, name = 'b1')
        self.w2= tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"),trainable=True, name = 'w2')
        self.b2 = tf.Variable(initial_value=b_init(shape=(input_dim, units), dtype="float32"), trainable=True, name = 'b2')
        self.custom_list = [4, 4, 0, 4, 4, 0, 2, 1, 0, 4, 1, 2, 1, 4, 2, 0, 1, 1, 2, 1, 1, 4, 0, 4, 2]

    def call(self, sem, mask):
        tempsem=tf.roll(sem, shift=-1, axis=1)
        t3=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1
        tempsem=tf.roll(sem, shift=1, axis=1)
        tempsem=tf.roll(tempsem, shift=1, axis=2)
        t6=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1
        tempsem=tf.roll(sem, shift=1, axis=1)
        tempsem=tf.roll(tempsem, shift=-1, axis=2)
        t7=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1
        tempsem=tf.roll(sem, shift=-1, axis=1)
        tempsem=tf.roll(tempsem, shift=1, axis=2)
        t8=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1
        tempsem=tf.roll(sem, shift=-1, axis=1)
        tempsem=tf.roll(tempsem, shift=-1, axis=2)
        t9=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1
        tempsem=tf.roll(sem, shift=-2, axis=1)
        t11=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=2, axis=2)
        t12=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-2, axis=2)
        t13=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=2, axis=1)
        tempsem=tf.roll(tempsem, shift=1, axis=2)
        t15=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=2, axis=1)
        tempsem=tf.roll(tempsem, shift=-1, axis=2)
        t16=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=2, axis=1)
        tempsem=tf.roll(tempsem, shift=-2, axis=2)
        t17=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-2, axis=1)
        tempsem=tf.roll(tempsem, shift=2, axis=2)
        t18=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-2, axis=1)
        tempsem=tf.roll(tempsem, shift=1, axis=2)
        t19=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-2, axis=1)
        tempsem=tf.roll(tempsem, shift=-1, axis=2)
        t20=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-2, axis=1)
        tempsem=tf.roll(tempsem, shift=-2, axis=2)
        t21=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=1, axis=1)
        tempsem=tf.roll(tempsem, shift=2, axis=2)
        t23=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        tempsem=tf.roll(sem, shift=-1, axis=1)
        tempsem=tf.roll( tempsem, shift=-2, axis=2)
        t25=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2
        sum = 0
        sum = sum + tf.keras.layers.Activation('sigmoid')(t3) + tf.keras.layers.Activation('sigmoid')(t6) + tf.keras.layers.Activation('relu')(t7) + tf.keras.layers.Activation('tanh')(t8) + tf.keras.layers.Activation('sigmoid')(t9) + tf.keras.layers.Activation('tanh')(t11) + tf.keras.layers.Activation('relu')(t12) + tf.keras.layers.Activation('tanh')(t13) + tf.keras.layers.Activation('relu')(t15) + tf.keras.layers.Activation('sigmoid')(t16) + tf.keras.layers.Activation('tanh')(t17) + tf.keras.layers.Activation('tanh')(t18) + tf.keras.layers.Activation('relu')(t19) + tf.keras.layers.Activation('tanh')(t20) + tf.keras.layers.Activation('tanh')(t21) + tf.keras.layers.Activation('sigmoid')(t23) + tf.keras.layers.Activation('relu')(t25)
        sum = tf.keras.layers.Activation('sigmoid')(sum)
        return sum + sem

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'w1': self.w1,
                'b1': self.b1,
                'w2': self.w2,
                'b2': self.b2,
            })
        return config   

def create_mainflow(input_size, model_name):
    inputs = Input(input_size)
    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    output = Conv2D(1, 1, activation = 'sigmoid')(x1)

    mainflow = Model(inputs=inputs, outputs=output, name=model_name) 
    return mainflow
def model_custom(args, pretrained_weights = None,input_size = (256,256,1)):
    input_size = (args.image_size, args.image_size, 1)
    mainflow = create_mainflow(input_size, 'mainflow')
    Mask1 = Input(input_size)
    SEM1 = mainflow(Mask1)
    Mask1_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(Mask1)
    SEM1_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(SEM1)
    Mask2 = Linear(units=256, input_dim=256)(SEM1_reshape, Mask1_reshape)
    Mask2_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(Mask2)
    SEM2=mainflow(Mask2_reshape)
    model = Model(inputs = Mask1, outputs = SEM2, name='model_custom')
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
