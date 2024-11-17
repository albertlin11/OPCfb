from model_custom import *
from data import *
import random
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import *

import argparse
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

def main(X):
    parser = argparse.ArgumentParser("main_two_repo")
    parser.add_argument('--GPU',type=str,default="0",help='GPU lebel') 
    parser.add_argument('--result_dir',type=str,default='test_result',help='path for result')
    parser.add_argument('--model',type=str,default='model_custom',help='model name')
    parser.add_argument('--epoch',type=int,default=3000,help='epoch') 
    parser.add_argument('--batch_size',type=int,default=2,help='batch_size')
    parser.add_argument('--edge_enhanced',type=float,default=10.0,help='edge_enhance factor')
    parser.add_argument('--seed',type=int,default=60,help='seed')
    parser.add_argument('--lr',type=float,default=1e-4,help='seed')
    parser.add_argument('--h_flip',type=bool,default=False,help='seed')
    parser.add_argument('--image_size',type=int,default=256,help='image_size')
    parser.add_argument('--custom_list',type=int,default = 0, nargs='+',help='custom_list')
    parser.add_argument('--trial',type=int,default=0,help='trial')
    parser.add_argument('--patience',type=int,default=20,help='patience')
    parser.add_argument('--mode', type=str, default="train", help='check if test only')

    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU 
    seed = args.seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    target_result_dir = "bceonly_0313" ## for test mode


    train_num = 168 ## check
    test_num = 39 ## check
    threshold = 0.2

    print("="*100)
    print("="*100)
    print("running the model: " + args.model)
    print("="*100)
    print("="*100)

    def soft_round(x, alpha=5.0, eps=1e-3):

        alpha_bounded = tf.maximum(alpha, eps)

        m = tf.floor(x) + .5
        r = x - m
        z = tf.tanh(alpha_bounded / 2.) * 2.
        y = m + tf.tanh(alpha_bounded * r) / z

        # For very low alphas, soft_round behaves like identity
        return tf.where(alpha < eps, x, y, name="soft_round")

    # function definition
    def my_loss(y_true, y_pred): # return modified_mse 
        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.numpy().copy())
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > 0.2, 1.0, mask)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > 0.2, 1.0, mask)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        score = tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred), mask))

        return score 

    def binary_mse(y_true, y_pred): # binary the pred before calculate mse
        # y_pred_binary = tf.where(y_pred >= 0.5, y_pred, 0.0)
        y_pred_binary = soft_round(y_pred)
        score = tf.reduce_mean(tf.square(y_true - y_pred_binary))
        return score

    def binary_bce(y_true, y_pred):


        # birarized y_pred
        # y_pred_binary = tf.keras.backend.round(y_pred)
        y_pred_binary = soft_round(y_pred)
        # clip
        y_pred_binary = tf.clip_by_value(y_pred_binary, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * tf.math.log(y_pred_binary)
        bce += (1 - y_true) * tf.math.log(1 - y_pred_binary)

        score = tf.reduce_mean(-bce)

        return score

    def loss_bce(y_true, y_pred):
        y_pred = np.clip(y_pred, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * np.log(y_pred)
        bce += (1 - y_true) * np.log(1 - y_pred)

        return -np.mean(bce)

    def loss_mse(y_true, y_pred): # normal mse

        difference = abs(y_pred - y_true)
        difference = np.multiply(difference, difference)
        score = np.mean(difference)

        return score

    def my_loss_bce(y_true, y_pred): # modified bce

        # mask = y_true.copy()
        # mask = np.reshape(mask, (test_num, 256, 256))
        # mask = np.where(mask > 0.2, 1.0, mask)
        # kernel_1 = np.ones((3,3),np.float32)/9
        # for i in range(test_num):
        #     mask[i] = cv2.filter2D(mask[i],-1,kernel_1)
        # mask = np.where(mask > 0.2, 1.0, mask)
        # mask = np.reshape(mask, (test_num, 256, 256, 1))
        # mask = mask * 10.0
        # mask = mask + 1.0

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > 0.2, 1.0, mask)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > 0.2, 1.0, mask)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        y_pred = np.clip(y_pred, 0 + 1e-7, 1 - 1e-7)
        
        bce = y_true * np.log(y_pred)
        bce += (1 - y_true) * np.log(1 - y_pred)
        bce = np.multiply(-bce, mask)

        return np.mean(bce)
                
    def my_loss_mse(y_true, y_pred): # modified mse

        # mask = y_true.copy()
        # mask = np.reshape(mask, (test_num, args.image_size, args.image_size))
        # mask = np.where(mask > 0.2, 1.0, mask)
        # kernel_1 = np.ones((3,3),np.float32)/9
        # for i in range(test_num):
        #     mask[i] = cv2.filter2D(mask[i],-1,kernel_1)
        # mask = np.where(mask > 0.2, 1.0, mask)
        # mask = np.reshape(mask, (test_num, args.image_size, args.image_size, 1))
        # mask = mask * args.edge_enhanced
        # mask = mask + 1.0

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > 0.2, 1.0, mask)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > 0.2, 1.0, mask)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        difference = abs(y_pred - y_true)
        difference = np.multiply(difference, difference)
        difference = np.multiply(difference, mask)
        score = np.mean(difference)

        return score

    def PA(y_true, y_pred): 
        y_pred = np.where(y_pred > threshold, 1.0, 0.0)
        # y_predXy_true0 = np.where(np.logical_and(y_pred == 0.0, y_true == 0.0), 1.0, 0.0) # binarized the value of y_predXy_true
        y_predXy_true1 = np.where(np.logical_and(y_pred == 1.0, y_true == 1.0), 1.0, 0.0) # binarized the value of y_predXy_true
        # print("y_predXy_true1: ", np.sum(y_predXy_true1))
        # print("y_predXy_true0: ", np.sum(y_predXy_true0))
        # acc = (np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0)) + np.sum(y_predXy_true0) / np.sum(np.where(y_true == 0.0, 1.0, 0.0))) / 2
        acc = np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0))
        return acc

    def pixelacc(y_true, y_pred): ## edge enhanced
        # the value of y_true is only 0 or 1
        # the value of y_pred is 0~1

        # for difference, we set y_pred to 0 or 1
        y_pred = np.where(y_pred > threshold, 1.0, 0.0) # binarized the value of y_pred
        
        # mask = y_true.copy() 
        # mask = np.reshape(mask, (test_num, args.image_size, args.image_size))
        # mask = np.where(mask > 0.2, 1.0, mask)
        # kernel_1 = np.ones((3,3),np.float32)/9
        # for i in range(test_num):
        #     mask[i] = cv2.filter2D(mask[i],-1,kernel_1)
        # mask = np.where(mask > 0.2, 1.0, mask)
        # mask = np.reshape(mask, (test_num, args.image_size, args.image_size, 1))
        # mask = mask * args.edge_enhanced
        # mask = mask + 1.0

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > 0.2, 1.0, mask)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > 0.2, 1.0, mask)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        # calculate the accuracy 
        difference = abs(y_pred - y_true) # it's the matrix of difference shape is (test_num, 256, 256, 1)

        # modified_difference = np.multiply(difference, mask) # shape is (test_num, 256, 256, 1)
        
        # y_pred U y_true
        y_predUy_true0 = np.where(np.logical_or(y_pred == 0.0, y_true == 0.0), 1.0, 0.0)
        y_predUy_true1 = np.where(np.logical_or(y_pred == 1.0, y_true == 1.0), 1.0, 0.0)
        y_predXy_true0 = np.where(np.logical_and(y_pred == 0.0, y_true == 0.0), 1.0, 0.0) # binarized the value of y_predXy_true
        y_predXy_true1 = np.where(np.logical_and(y_pred == 1.0, y_true == 1.0), 1.0, 0.0) # binarized the value of y_predXy_true
        print("y_predXy_true1: ", np.sum(y_predXy_true1))
        print("y_predUy_true1: ", np.sum(y_predUy_true1))
        print("y_predXy_true0: ", np.sum(y_predXy_true0))
        print("y_predUy_true0: ", np.sum(y_predUy_true0))

        # IOU = (np.sum(y_predXy_true1) / np.sum(y_predUy_true1) + np.sum(y_predXy_true0) / np.sum(y_predUy_true0)) / 2
        # PA = (np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0)) + np.sum(y_predXy_true0) / np.sum(np.where(y_true == 0.0, 1.0, 0.0))) / 2
        # modified_PA = (np.sum(np.multiply(y_predXy_true1, mask)) / np.sum(np.multiply(np.where(y_true == 1.0, 1.0, 0.0), mask)) + np.sum(np.multiply(y_predXy_true0, mask)) / np.sum(np.multiply(np.where(y_true == 0.0, 1.0, 0.0), mask))) / 2
        
        IOU = (np.sum(y_predXy_true1) / np.sum(y_predUy_true1))
        PA = np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0))
        modified_PA = np.sum(np.multiply(y_predXy_true1, mask)) / np.sum(np.multiply(np.where(y_true == 1.0, 1.0, 0.0), mask)) 

        difference_picture = np.where(difference >= 0.5, 255, 0.0) ## depend on the difference, set to black(0) or white(255)


        return PA, modified_PA, IOU, difference_picture

    def SaveTestLossData(path, results, test_loss, history=None, loss_fun="default"):
        # record the result of training with BCE
        filename = path + "/record_" + loss_fun
        f = open(filename + ".txt", "w")

        # save test loss
        y_pred = results.copy()
        test_loss_bce = loss_bce(y_true, y_pred)
        f.write("test_loss_bce" + '\n')
        f.write(str(test_loss_bce) + '\n')

        # save test loss with MSE
        y_pred = results.copy()
        test_loss_mse = loss_mse(y_true, y_pred)
        f.write("test_loss_mse" + '\n')
        f.write(str(test_loss_mse) + '\n')

        # save test loss with modified BCE
        y_pred = results.copy()
        test_my_loss_bce = my_loss_bce(y_true, y_pred)
        f.write("test_modified_loss_bce" + '\n')
        f.write(str(test_my_loss_bce) + '\n')

        # save test loss with modified MSE
        y_pred = results.copy()
        test_my_loss_mse = my_loss_mse(y_true, y_pred)
        f.write("test_modified_loss_mse" + '\n')
        f.write(str(test_my_loss_mse) + '\n')

        # save difference
        y_pred = results.copy()
        pixelaccuracy, modified_pixelaccuracy, IOU, diffrence_pic = pixelacc(y_true, y_pred)
        SaveFigureBy0to255(path, diffrence_pic, loss_fun ) ## save black white image
        f.write("pixelacc" + '\n')
        f.write(str(pixelaccuracy) + '\n')
        f.write("modified_pixelacc" + '\n')
        f.write(str(modified_pixelaccuracy) + '\n')

        ## the flowing one is for ga
        f.write("val_my_loss" + '\n')
        f.write(str(min(history.history['val_my_loss'])) + '\n')

        f.write("IOU" + '\n')
        f.write(str(IOU) + '\n')
        f.write("test_loss_" + loss_fun + '\n')
        f.write(str(test_loss[0]) + '\n')
        f.write("test_loss_metrics_" + loss_fun + '\n')
        f.write(str(test_loss[1]) + '\n')
        f.close()

        if (history != None):
            # record the result of training process
            hist_df = pd.DataFrame(history.history) 

            # save to json:  
            hist_json_file = path + '/history_' + loss_fun +  '.json' 
            with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f)

            # or save to csv: 
            hist_csv_file = path + '/history_' + loss_fun +  '.csv' 
            with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)


    # choose model
    if(args.model == "unet"):
        print("unet")
        model = models.unet_2d((args.image_size, args.image_size, 1), [64, 128, 256, 512, 1024], n_labels=1,
                            stack_num_down=2, stack_num_up=1,
                            activation='GELU', output_activation='Sigmoid', 
                            batch_norm=True, pool='max', unpool='nearest', name='unet')
    elif(args.model == "unet_cross"):
        print("unet_cross")
        model = models.att_unet_2d((args.image_size, args.image_size, 1), [64, 128, 256, 512], n_labels=1,
                                stack_num_down=2, stack_num_up=2,
                                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                                batch_norm=True, pool=False, unpool='bilinear', name='attunet')
    elif(args.model == "unet_self"):
        print("unet_self")
        model = models.satt_unet_2d((args.image_size, args.image_size, 1), [64, 128, 256, 512], n_labels=1,
                                stack_num_down=2, stack_num_up=2,
                                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                                batch_norm=True, pool=False, unpool='bilinear', name='attunet')
    elif(args.model == "small_unet"):
        print("small_unet")
        model = models.unet_2d((args.image_size, args.image_size, 1), [8, 32, 64, 128, 160], n_labels=1,
                            stack_num_down=2, stack_num_up=1,
                            activation='GELU', output_activation='Sigmoid', 
                            batch_norm=True, pool='max', unpool='nearest', name='unet')
    elif(args.model == "small_unet_cross"):
        print("small_unet_cross")
        model = models.att_unet_2d((args.image_size, args.image_size, 1), [32, 64, 96, 128], n_labels=1,
                                stack_num_down=2, stack_num_up=2,
                                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                                batch_norm=True, pool=False, unpool='bilinear', name='attunet')
    elif(args.model == "small_unet_self"):
        print("small_unet_self")
        model = models.satt_unet_2d((args.image_size, args.image_size, 1), [32, 64, 96, 128], n_labels=1,
                                stack_num_down=2, stack_num_up=2,
                                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                                batch_norm=True, pool=False, unpool='bilinear', name='attunet')
    elif(args.model == "prof_input_1_feedback"):
        print("prof_input_1_feedback")
        model = prof_input_1_feedback(args)
    elif(args.model == "prof_input_1_feedback_cross_a"):
        print("prof_input_1_feedback_cross")
        model = prof_input_1_feedback_cross_a(args)
    elif(args.model == "prof_input_1_feedback_self_a"):
        print("prof_input_1_feedback_self")
        model = prof_input_1_feedback_self_a(args)
    elif(args.model == "model_base"):
        print("model_base")
        model = model_base(args)
    elif(args.model == "model_custom"):
        print("model_custom")
        model = model_custom(args)
    else:
        print("please check args.model")

    # set loss
    first_loss = "binary_crossentropy"
    second_loss = binary_mse

    # setting the path
    if args.mode == "train":
        path = args.result_dir + '/' + args.model + '_'  + str(args.lr) + '_' + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(X)  + '_' + str(args.patience)
    elif args.mode == "test":
        path = args.result_dir + '/' + args.model + '_'  + str(args.lr) + '_' + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(X)  + '_' + str(args.patience) + '_test'
        test_path = target_result_dir + '/' + args.model + '_'  + str(args.lr) + '_' + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(args.custom_list) + '_' + str(args.trial) + '_' + str(args.patience) 

    # check if the data has been generated
    if not os.path.isdir(path): # if the path is not exist, create the path
        os.makedirs(path)
    else: # if the path is exist, check the record.txt and return the value of val_my_loss
        print("The custom_list has been tested")
        path_1 = path
        path_2 = 'record_binary_mse.txt'
        tmp = []

        with open(path_1 + '/' + path_2, 'r') as f:
            for line in f.read().splitlines():            
                tmp.append(line)
            print(float(tmp[9]))
        return float(tmp[9])

    print("path: ", path)

    # the name of model
    model_name = args.model + '_' + str(args.lr) + '_'  + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(args.custom_list) + '_' + str(args.trial) + '_' + str(args.patience) + '.hdf5'

    # choose mode        
    if(args.mode == "train"):
        # save summary
        # save summary
        summary_file = path + '/summary.txt'

        with open(summary_file, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.close()

        # load training data and test data
        x_train_dir = os.getcwd() + "/dataset/train/x_png"
        x_trainData = testGenerator(x_train_dir, num_image=train_num, target_size = (args.image_size, args.image_size))
        x_train = np.array(list(x_trainData))
        y_train_dir = os.getcwd() + "/dataset/train/y_png"
        y_trainData = testGenerator(y_train_dir, num_image=train_num, target_size = (args.image_size, args.image_size))
        y_train = np.array(list(y_trainData)) # the value of y_train is only 0 or 1

        x_test_dir = os.getcwd() + "/dataset/test/x_png"
        x_testData = testGenerator(x_test_dir, num_image=test_num, target_size = (args.image_size, args.image_size))
        x_test = np.array(list(x_testData))
        y_test_dir = os.getcwd() + "/dataset/test/y_png"
        y_testData = testGenerator(y_test_dir, num_image=test_num, target_size = (args.image_size, args.image_size))
        y_true = np.array(list(y_testData)) # the value of y_true is only 0 or 1

        # save test mask if not exist
        if not os.path.exists('test_mask'):
            os.makedirs('test_mask')
            num = y_true.shape[0]
            mask = tf.convert_to_tensor(y_true.copy())
            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
            mask = tf.where(mask > 0.2, 1.0, mask)
            kernel_1 = np.ones((3, 3), np.float32) / 9
            kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

            mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
            mask = tf.where(mask > 0.2, 1.0, mask)
            mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
            for i in range(num):
                
                cv2.imwrite('test_mask/' + str(i).zfill(3) + '.png', mask[i].numpy() * 255)


        model.compile(optimizer = Adam(learning_rate = args.lr), loss = binary_bce, metrics = [my_loss, PA], run_eagerly=True)
        model_name_bce = model_name[:-5] + "_bce.hdf5"
        model_checkpoint = ModelCheckpoint(path + '/' + model_name_bce, monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode="min", save_freq="epoch",)
        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True, mode="min")
        print("*"*100)
        print("*"*100)
        print("train BCE start!")
        print("*"*100)
        print("*"*100)
        history = model.fit(x_train,y_train, validation_split=0.2,batch_size=args.batch_size,epochs=args.epoch,shuffle=True,callbacks=[early_cb, model_checkpoint])
        print("*"*100)
        print("*"*100)
        print("Done training with BCE!")
        print("*"*100)
        print("*"*100)

        if model.loss == "binary_crossentropy" or model.loss == "mean_squared_error":
            model_loss_name = model.loss
        else:
            model_loss_name = str(model.loss.__name__)

        # plot training loss and val_loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # legend
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path + '/loss_' + model_loss_name + '.png')
        plt.close()

        # plot pixel accuracy
        plt.plot(history.history['PA'])
        plt.plot(history.history['val_PA'])
        plt.title('model pixel accuracy')
        plt.ylabel('pixel accuracy')
        plt.xlabel('epoch')
        # legend
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path + '/PA_' + model_loss_name + '.png')
        plt.close()



        # predict and evaluate with BCE
        model.load_weights(path + '/' + model_name_bce) # load weight is for prediction
        results = model.predict(x_test, verbose=1)
        results_binary = np.where(results >= threshold, 1.0, 0.0)
        saveResult(path, results_binary, loss_fun=model_loss_name + '_binaried')
        saveResult(path, results, loss_fun=model_loss_name) 
        test_loss = model.evaluate(x = x_test, y = y_true, batch_size = args.batch_size) # test loss with BCE
        # save training loss and val_loss
        SaveTestLossData(path, results, test_loss, history, loss_fun=model_loss_name)




        
        # get the middle layer picture
        if(args.model == "prof_input_1_feedback" or args.model == "prof_input_1_feedback_cross_a" or args.model == "prof_input_1_feedback_self_a" or args.model == "model_custom"):
            input_layer = model.get_layer('mainflow').input
            output_layer = model.get_layer('mainflow').output
            middle_layer = Model(inputs=input_layer, outputs=output_layer, name="middle_layer_" + args.model)
            middle_outputs = middle_layer.predict(x_test)
            middle_layer.summary()
        
            # save to png
            for i in range(middle_outputs.shape[0]):
                saveResult(path, middle_outputs, loss_fun="mid")  

    return min(history.history['val_my_loss'])

