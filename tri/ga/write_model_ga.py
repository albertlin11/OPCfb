def write_file(custom_list):
    print("write: custom_list")
    print(custom_list)
    fname = 'model_custom.py'
    with open(fname, 'w') as f:
        f.write("import numpy as np" + "\n" + "import tensorflow as tf" + "\n" + "from tensorflow.keras.models import *" + "\n" + "from tensorflow.keras.layers import *" + "\n" + "from tensorflow.keras.optimizers import *" + "\n")
        f.write("class Linear(tf.keras.layers.Layer):" + "\n")
        f.write("    def __init__(self, units=32, input_dim=32):" + "\n")
        f.write("        super().__init__()" + "\n")
        f.write("        w_init = tf.random_normal_initializer()" + "\n")
        f.write("        b_init = tf.zeros_initializer()" + "\n")
        f.write("        self.w1 = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype=\"float32\"), trainable=True, name = 'w1')" + "\n")
        f.write("        self.b1 = tf.Variable(initial_value=b_init(shape=(input_dim, units), dtype=\"float32\"), trainable=True, name = 'b1')" + "\n")
        f.write("        self.w2= tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype=\"float32\"),trainable=True, name = 'w2')" + "\n")
        f.write("        self.b2 = tf.Variable(initial_value=b_init(shape=(input_dim, units), dtype=\"float32\"), trainable=True, name = 'b2')" + "\n")
        f.write("        self.custom_list = [")

        for i in range(len(custom_list)):
            if(i != len(custom_list) - 1):
                f.write(str(int(custom_list[i])) + ", ")
            elif(i == len(custom_list) - 1):
                f.write(str(int(custom_list[i])) + "]\n")
        f.write("\n")
        
        f.write("    def call(self, sem, mask):" + "\n")
        if(custom_list[0] != 4):
            f.write("        t1=tf.multiply(mask, tf.matmul(sem, self.w1)) + self.b1" + "\n")
        if(custom_list[1] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=1)" + "\n")
            f.write("        t2=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[2] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=1)" + "\n")
            f.write("        t3=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[3] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=2)" + "\n")
            f.write("        t4=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[4] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=2)" + "\n")
            f.write("        t5=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[5] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=1, axis=2)" + "\n")
            f.write("        t6=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[6] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-1, axis=2)" + "\n")
            f.write("        t7=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[7] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=1, axis=2)" + "\n")
            f.write("        t8=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[8] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-1, axis=2)" + "\n")
            f.write("        t9=tf.multiply(mask, tf.matmul(tempsem, self.w1)) + self.b1" + "\n")
        if(custom_list[9] != 4):       
            f.write("        tempsem=tf.roll(sem, shift=2, axis=1)" + "\n")
            f.write("        t10=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[10] != 4):        
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=1)" + "\n")
            f.write("        t11=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[11] != 4):       
            f.write("        tempsem=tf.roll(sem, shift=2, axis=2)" + "\n")
            f.write("        t12=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[12] != 4):      
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=2)" + "\n")
            f.write("        t13=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[13] != 4):
            f.write("        tempsem=tf.roll(sem, shift=2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=2, axis=2)" + "\n")
            f.write("        t14=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[14] != 4):
            f.write("        tempsem=tf.roll(sem, shift=2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=1, axis=2)" + "\n")
            f.write("        t15=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[15] != 4):
            f.write("        tempsem=tf.roll(sem, shift=2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-1, axis=2)" + "\n")
            f.write("        t16=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[16] != 4):
            f.write("        tempsem=tf.roll(sem, shift=2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-2, axis=2)" + "\n")
            f.write("        t17=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[17] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=2, axis=2)" + "\n")
            f.write("        t18=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[18] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=1, axis=2)" + "\n")
            f.write("        t19=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[19] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-1, axis=2)" + "\n")
            f.write("        t20=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[20] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-2, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-2, axis=2)" + "\n")
            f.write("        t21=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[21] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=2, axis=2)" + "\n")
            f.write("        t22=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[22] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=2, axis=2)" + "\n")
            f.write("        t23=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[23] != 4):
            f.write("        tempsem=tf.roll(sem, shift=1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll(tempsem, shift=-2, axis=2)" + "\n")
            f.write("        t24=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
        if(custom_list[24] != 4):
            f.write("        tempsem=tf.roll(sem, shift=-1, axis=1)" + "\n")
            f.write("        tempsem=tf.roll( tempsem, shift=-2, axis=2)" + "\n")
            f.write("        t25=tf.multiply(mask, tf.matmul(tempsem, self.w2)) + self.b2" + "\n")
     
        f.write("        sum = 0" + "\n")  
        f.write("        sum = sum + ")  

        activation_list=['sigmoid', 'tanh', 'relu', 'softmax']
        for i in range(len(custom_list)):
            if(custom_list[i] != 4):
                last = i
        for i in range(len(custom_list)):
            if(custom_list[i] != 4 and i != last):
                 f.write("tf.keras.layers.Activation('" + activation_list[int(custom_list[i])] + "')(t" + str(i + 1) + ") + ")
            elif(custom_list[i] != 4 and i == last):
                 f.write("tf.keras.layers.Activation('" + activation_list[int(custom_list[i])] + "')(t" + str(i + 1) + ")" + "\n")
        f.write("        sum = tf.keras.layers.Activation('sigmoid')(sum)" + "\n")  
        f.write("        return sum + sem" + "\n")  
        f.write("\n")
    
        f.write("    def get_config(self):" + "\n")
        f.write("        config = super().get_config().copy()" + "\n")
        f.write("        config.update({" + "\n")
        f.write("                'w1': self.w1," + "\n")
        f.write("                'b1': self.b1," + "\n")
        f.write("                'w2': self.w2," + "\n")
        f.write("                'b2': self.b2," + "\n")
        f.write("            })" + "\n")
        f.write("        return config   " + "\n")

        f.write("\n")

        f.write("def create_mainflow(input_size, model_name):" + "\n")
        f.write("    inputs = Input(input_size)" + "\n")
        f.write("    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)" + "\n")
        f.write("    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)" + "\n")
        f.write("    x1=Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)" + "\n")
        f.write("    x1=Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)" + "\n")
        f.write("    x1=Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)" + "\n")
        f.write("    x1 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)" + "\n")
        f.write("    output = Conv2D(1, 1, activation = 'sigmoid')(x1)" + "\n" + "\n")
        f.write("    mainflow = Model(inputs=inputs, outputs=output, name=model_name) " + "\n")
        f.write("    return mainflow" + "\n")



        f.write("def model_custom(args, pretrained_weights = None,input_size = (256,256,1)):" + "\n")
        f.write("    input_size = (args.image_size, args.image_size, 1)" + "\n")
        f.write("    mainflow = create_mainflow(input_size, 'mainflow')" + "\n")
        f.write("    Mask1 = Input(input_size)" + "\n")
        f.write("    SEM1 = mainflow(Mask1)" + "\n")
        f.write("    Mask1_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(Mask1)" + "\n")
        f.write("    SEM1_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(SEM1)" + "\n")
        f.write("    Mask2 = Linear(units=256, input_dim=256)(SEM1_reshape, Mask1_reshape)" + "\n")
        f.write("    Mask2_reshape=tf.keras.layers.Reshape((256, 256), input_shape=(256,256,1))(Mask2)" + "\n")
        f.write("    SEM2=mainflow(Mask2_reshape)" + "\n")
        f.write("    model = Model(inputs = Mask1, outputs = SEM2, name='model_custom')" + "\n")
   
        f.write("    if(pretrained_weights):" + "\n")
        f.write("        model.load_weights(pretrained_weights)" + "\n")
        f.write("    return model" + "\n")
        print("The model is done by written")
    f.close()

