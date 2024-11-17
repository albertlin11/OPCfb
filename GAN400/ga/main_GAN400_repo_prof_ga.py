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

    parser = argparse.ArgumentParser("main_GAN400_repo")
    parser.add_argument('--GPU',type=str,default="1",help='GPU lebel') 
    parser.add_argument('--result_dir',type=str,default='test_result',help='path for result')
    parser.add_argument('--model',type=str,default='model_custom',help='model name')
    parser.add_argument('--epoch',type=int,default=2,help='epoch') 
    parser.add_argument('--batch_size',type=int,default=2,help='batch_size')
    parser.add_argument('--edge_enhanced',type=float,default=10.0,help='edge_enhance factor')
    parser.add_argument('--seed',type=int,default=60,help='seed')
    parser.add_argument('--lr',type=float,default=1e-4,help='seed')
    parser.add_argument('--h_flip',type=bool,default=False,help='seed')
    parser.add_argument('--image_size',type=int,default=256,help='image_size')
    parser.add_argument('--custom_list',type=int,default = 0, nargs='+',help='custom_list')
    parser.add_argument('--trial',type=int,default=0,help='trial')
    parser.add_argument('--patience',type=int,default=20,help='patience')
    parser.add_argument('--augmode', type=str, default="none", help='check if aug only')
    parser.add_argument('--augrotate', type=float, default=0.0, help='augmentation rotate')
    parser.add_argument('--augtranslation', type=float, default=0.0, help='augmentation translation')
    parser.add_argument('--mode', type=str, default="train", help='train or test')
    parser.add_argument('--version', type=int, default=0, help='version') # used to differ same training condition but different main.py

    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU 
    seed = args.seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



    train_num = 320 ## check
    test_num = 80 ## check
    threshold = 0.2
    mask_threshold = 0.2
    # backword = "_lossmbbce_th0.2_mth0.2" # used to check if the model is already trained before

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
    

    def binary_mse(y_true, y_pred): # binary the pred before calculate mse
        y_pred_binary = soft_round(y_pred)
        score = tf.reduce_mean(tf.square(y_true - y_pred_binary))
        return score

    def binary_bce(y_true, y_pred):


        # birarized y_pred
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

    def loss_mbce(y_true, y_pred): # modified bce

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        y_pred = np.clip(y_pred, 0 + 1e-7, 1 - 1e-7)
        
        bce = y_true * np.log(y_pred)
        bce += (1 - y_true) * np.log(1 - y_pred)
        bce = np.multiply(-bce, mask)

        return np.mean(bce)
                
    def loss_mmse(y_true, y_pred): # modified mse

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        difference = abs(y_pred - y_true)
        difference = np.multiply(difference, difference)
        difference = np.multiply(difference, mask)
        score = np.mean(difference)

        return score

    def loss_bbce(y_true, y_pred): # soft_round the pred before calculate bce
        # birarized y_pred
        # y_pred_binary = tf.keras.backend.round(y_pred)
        y_pred_binary = soft_round(y_pred)
        # clip
        y_pred_binary = tf.clip_by_value(y_pred_binary, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * tf.math.log(y_pred_binary)
        bce += (1 - y_true) * tf.math.log(1 - y_pred_binary)

        score = tf.reduce_mean(-bce)


        return score.numpy()

    def loss_mbbce(y_true, y_pred): # modified bce

        num = y_true.shape[0]

        # if y_true have numpy:
        if (type(y_true) == np.ndarray):
            mask = tf.convert_to_tensor(y_true.copy())
        else:
            mask = tf.convert_to_tensor(y_true.numpy().copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0

        y_pred_binary = soft_round(y_pred)
        y_pred_binary = tf.clip_by_value(y_pred_binary, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * tf.math.log(y_pred_binary)
        bce += (1 - y_true) * tf.math.log(1 - y_pred_binary)
        bce = tf.multiply(-bce, mask)

        return tf.reduce_mean(bce)

    def test_bbce(y_true, y_pred): # binary the pred before calculate bce
        # birarized y_pred
        # y_pred = soft_round(y_pred)
        y_pred_binary = np.where(y_pred > threshold, 1.0, 0.0)
        # clip
        y_pred_binary = np.clip(y_pred_binary, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * np.log(y_pred_binary)
        bce += (1 - y_true) * np.log(1 - y_pred_binary)

        score = -np.mean(bce)

        return score

    def test_mbbce(y_true, y_pred): # modified bce

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0
        # y_pred = soft_round(y_pred)
        y_pred_binary = np.where(y_pred > threshold, 1.0, 0.0)
        y_pred_binary = np.clip(y_pred_binary, 0 + 1e-7, 1 - 1e-7)
        bce = y_true * np.log(y_pred_binary)
        bce += (1 - y_true) * np.log(1 - y_pred_binary)
        bce = np.multiply(-bce, mask)

        return np.mean(bce)

    def test_bmse(y_true, y_pred):
        # y_pred = soft_round(y_pred)
        y_pred = np.where(y_pred > threshold, 1.0, 0.0)
        score = np.mean(np.square(y_true - y_pred))
        return score

        
    def test_mbmse(y_true, y_pred): #  modified binary mse

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = mask * args.edge_enhanced
        mask = mask + 1.0
        # y_pred = soft_round(y_pred)
        y_pred = np.where(y_pred > threshold, 1.0, 0.0)
        difference = np.multiply(y_true - y_pred, y_true - y_pred)
        difference = np.multiply(difference, mask)
        score = np.mean(difference)

        return score



        


    def PA(y_true, y_pred): 
        y_pred = np.where(y_pred > threshold, 1.0, 0.0)
        y_predXy_true0 = np.where(np.logical_and(y_pred == 0.0, y_true == 0.0), 1.0, 0.0) # binarized the value of y_predXy_true
        y_predXy_true1 = np.where(np.logical_and(y_pred == 1.0, y_true == 1.0), 1.0, 0.0) # binarized the value of y_predXy_true
        # print("y_predXy_true1: ", np.sum(y_predXy_true1))
        # print("y_predXy_true0: ", np.sum(y_predXy_true0))
        acc = (np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0)) + np.sum(y_predXy_true0) / np.sum(np.where(y_true == 0.0, 1.0, 0.0))) / 2
        # acc = np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0))
        return acc

    def pixelacc(y_true, y_pred): ## edge enhanced
        # the value of y_true is only 0 or 1
        # the value of y_pred is 0~1

        # for difference, we set y_pred to 0 or 1
        y_pred = np.where(y_pred > threshold, 1.0, 0.0) # binarized the value of y_pred

        num = y_true.shape[0]
        mask = tf.convert_to_tensor(y_true.copy())
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (num, args.image_size, args.image_size, 1))
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
        kernel_1 = np.ones((3, 3), np.float32) / 9
        kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

        mask = tf.nn.conv2d(mask, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
        mask = tf.where(mask > mask_threshold, 1.0, 0.0)
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

        IOU = (np.sum(y_predXy_true1) / np.sum(y_predUy_true1) + np.sum(y_predXy_true0) / np.sum(y_predUy_true0)) / 2
        PA = (np.sum(y_predXy_true1) / np.sum(np.where(y_true == 1.0, 1.0, 0.0)) + np.sum(y_predXy_true0) / np.sum(np.where(y_true == 0.0, 1.0, 0.0))) / 2
        modified_PA = (np.sum(np.multiply(y_predXy_true1, mask)) / np.sum(np.multiply(np.where(y_true == 1.0, 1.0, 0.0), mask)) + np.sum(np.multiply(y_predXy_true0, mask)) / np.sum(np.multiply(np.where(y_true == 0.0, 1.0, 0.0), mask))) / 2

        difference_picture = np.where(difference >= 0.5, 255, 0.0) ## depend on the difference, set to black(0) or white(255)


        return PA, modified_PA, IOU, difference_picture



    def SaveTestLossData(path, results, history=None, loss_fun="default"):
        # record the result of training with BCE
        filename = path + "/record_" + loss_fun
        f = open(filename + ".txt", "w")

        # save test loss
        y_pred = results.copy()
        loss_bce_score = loss_bce(y_true, y_pred)
        f.write("test_bce" + '\n')
        f.write(str(loss_bce_score) + '\n')

        # save test loss with MSE
        y_pred = results.copy()
        loss_mse_score = loss_mse(y_true, y_pred)
        f.write("test_mse" + '\n')
        f.write(str(loss_mse_score) + '\n')

        # save test loss with modified BCE
        y_pred = results.copy()
        loss_mbce_score = loss_mbce(y_true, y_pred)
        f.write("test_mbce" + '\n')
        f.write(str(loss_mbce_score) + '\n')

        # save test loss with modified MSE
        y_pred = results.copy()
        loss_mmse_score = loss_mmse(y_true, y_pred)
        f.write("test_mmse" + '\n')
        f.write(str(loss_mmse_score) + '\n')

        # save difference
        y_pred = results.copy()
        pixelaccuracy, modified_pixelaccuracy, IOU, diffrence_pic = pixelacc(y_true, y_pred)
        SaveFigureBy0to255(path, diffrence_pic, loss_fun ) ## save black white image
        f.write("test_PA" + '\n')
        f.write(str(pixelaccuracy) + '\n')
        f.write("test_mPA" + '\n')
        f.write(str(modified_pixelaccuracy) + '\n')
        f.write("test_IOU" + '\n')
        f.write(str(IOU) + '\n')

        # save BBCE 
        y_pred = results.copy()
        loss_bbce_score = loss_bbce(y_true, y_pred)
        f.write("loss_bbce" + '\n')
        f.write(str(loss_bbce_score) + '\n')

        # save modified BBCE
        y_pred = results.copy()
        loss_mbbce_score = loss_mbbce(y_true, y_pred)
        f.write("loss_mbbce" + '\n')
        f.write(str(loss_mbbce_score.numpy()) + '\n') 


        # save training loss bbce
        f.write("train_loss_bbce" + '\n')
        f.write(str(history.history['loss'][-1]) + '\n')

        # save training mbbce
        f.write("train_val_loss_bbce" + '\n')
        f.write(str(history.history['val_loss'][-1]) + '\n')

        # save training loss mbbce
        f.write("train_loss_mbbce" + '\n')
        f.write(str(history.history['loss_mbbce'][-1]) + '\n')

        # save training val loss mbbce
        f.write("train_val_loss_mbbce" + '\n')
        f.write(str(history.history['val_loss_mbbce'][-1]) + '\n')

        # save training PA
        f.write("train_PA" + '\n')
        f.write(str(history.history['PA'][-1]) + '\n')

        # save training val PA
        f.write("train_val_PA" + '\n')
        f.write(str(history.history['val_PA'][-1]) + '\n')

        # save test bmse
        y_pred = results.copy()
        f.write("test_bmse" + '\n')
        test_bmse_score = test_bmse(y_true, y_pred)
        f.write(str(test_bmse_score) + '\n')

        # save test_mbmse
        y_pred = results.copy()
        f.write("test_mbmse" + '\n')
        test_mbmse_score = test_mbmse(y_true, y_pred)
        f.write(str(test_mbmse_score) + '\n')

        # save test_bbce
        y_pred = results.copy()
        f.write("test_bbce" + '\n')
        test_bbce_score = test_bbce(y_true, y_pred)
        f.write(str(test_bbce_score) + '\n')

        # save test_mbbce
        y_pred = results.copy()
        f.write("test_mbbce" + '\n')
        test_mbbce_score = test_mbbce(y_true, y_pred)
        f.write(str(test_mbbce_score) + '\n')

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

    model = model_custom(args)



    # setting the path
    result_filename = args.model + '_'  + str(args.lr) + '_' + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(args.custom_list) + '_' + str(args.trial) + '_' + str(args.patience) + '_' + str(X)
    path = args.result_dir + '/' + result_filename


    # path is the path of result
    if not os.path.isdir(path):
        os.makedirs(path)

    print("path: ", path)

    # the name of model
    model_name = args.model + '_' + str(args.lr) + '_'  + str(args.epoch) + '_' + str(args.batch_size) + '_' + str(args.edge_enhanced) + '_' + str(args.seed) + '_' + str(args.custom_list) + '_' + str(args.trial) + '_' + str(args.patience) + '.hdf5'

    summary_file = path + '/summary.txt'

    with open(summary_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    # model compile
    model.compile(optimizer = Adam(learning_rate = args.lr), loss = binary_bce, metrics = [loss_mbbce, PA], run_eagerly=True)


    # set model_name 
    if model.loss == "binary_crossentropy" or model.loss == "mean_squared_error":
        model_loss_name = model.loss
    else:
        model_loss_name = str(model.loss.__name__)




    ## augmentation---------------------------------------------------------------------------

    x_data_augmentation = tf.keras.Sequential([

    # # rotate
    tf.keras.layers.experimental.preprocessing.RandomRotation(
        factor=args.augrotate,
        fill_mode='constant',
        seed=args.seed,
        fill_value=0.0,
        ),

    # translation
    tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=args.augtranslation,
        width_factor=args.augtranslation,
        fill_mode='constant',
        seed=args.seed,
        fill_value=0.0,
        ),

    ])

    y_data_augmentation = tf.keras.Sequential([

    # # rotate
    tf.keras.layers.experimental.preprocessing.RandomRotation(
        factor=args.augrotate,
        fill_mode='constant',
        seed=args.seed,
        fill_value=0.0,
        ),

    # translation
    tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=args.augtranslation,
        width_factor=args.augtranslation,
        fill_mode='constant',
        seed=args.seed,
        fill_value=0.0,
        ),

    ])
    ## augmentation---------------------------------------------------------------------------


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



    if args.augmode == "augboth":
        
        x_train_origin = x_train.copy()
        y_train_origin = y_train.copy()
        x_test_origin = x_test.copy()
        y_true_origin = y_true.copy()
        for i in range(1):
            x_train_aug = x_data_augmentation(x_train_origin)
            x_train_aug = np.where(x_train_aug >= 0.5, 1.0, 0.0)
            x_train = np.concatenate((x_train, x_train_aug), axis=0)

            y_train_aug = y_data_augmentation(y_train_origin)
            y_train_aug = np.where(y_train_aug >= 0.5, 1.0, 0.0)
            y_train = np.concatenate((y_train, y_train_aug), axis=0)

        x_test_aug = x_data_augmentation(x_test_origin)
        x_test_aug = np.where(x_test_aug >= 0.5, 1.0, 0.0)
        x_test = np.concatenate((x_test, x_test_aug), axis=0)

        y_true_aug = y_data_augmentation(y_true_origin)
        y_true_aug = np.where(y_true_aug >= 0.5, 1.0, 0.0)
        y_true = np.concatenate((y_true, y_true_aug), axis=0)





    elif args.augmode == "augxonly":

        x_train_origin = x_train.copy()
        x_test_origin = x_test.copy()
        y_train_origin = y_train.copy()


        for i in range(8):
            x_train_aug = x_data_augmentation(x_train_origin)
            x_train_aug = np.where(x_train_aug >= 0.5, 1.0, 0.0)
            x_train = np.concatenate((x_train, x_train_aug), axis=0)

            y_train = np.concatenate((y_train, y_train_origin), axis=0)

        x_test_aug = x_data_augmentation(x_test_origin)
        x_test_aug = np.where(x_test_aug >= 0.5, 1.0, 0.0)
        x_test = np.concatenate((x_test, x_test_aug), axis=0)
        y_true = np.concatenate((y_true, y_true), axis=0)



    else:
        print("no augmentation")



    # save x_train and y_train, x_test and y_true for checking
    if not os.path.isdir(path + '/data_check'):
        os.makedirs(path + '/data_check')

    ## get data of training data 
    num = y_true.shape[0]
    mask_figure = tf.convert_to_tensor(y_true.copy())
    mask_figure = tf.cast(mask_figure, tf.float32)
    mask_figure = tf.reshape(mask_figure, (num, args.image_size, args.image_size, 1))
    mask_figure = tf.where(mask_figure > mask_threshold, 1.0, 0.0)
    kernel_1 = np.ones((3, 3), np.float32) / 9
    kernel_1_resize = tf.reshape(kernel_1, (3, 3, 1, 1)) #(filter_height, filter_width, in_channels, out_channels)

    mask_figure = tf.nn.conv2d(mask_figure, kernel_1_resize, strides=[1, 1, 1, 1], padding='SAME')
    mask_figure = tf.where(mask_figure > mask_threshold, 1.0, 0.0)


    saveResult(path + '/data_check', x_train, loss_fun="xtrain")
    saveResult(path + '/data_check', y_train, loss_fun="ytrain")
    saveResult(path + '/data_check', x_test, loss_fun="xtest")
    saveResult(path + '/data_check', y_true, loss_fun="ytest")
    saveResult(path + '/data_check', mask_figure, loss_fun="mask")


    model_name_bce = model_name[:-5] + "_bce.hdf5"

    model_path = path + '/' + model_name_bce
    #================================================================================================
    #                                       train mode
    #================================================================================================

    if args.mode == "train":
        print("model_path: ", model_path)
        if os.path.isfile(model_path):
            model.load_weights(model_path)
            print("="*100)
            print("load the model weight")
            print("="*100)


        model_checkpoint = ModelCheckpoint(path + '/' + model_name_bce, monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode="min", save_freq="epoch",)
        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,restore_best_weights=True, mode="min")
        print("="*100)
        print("M" + args.augmode + "_T" + str(args.augtranslation) + '_R' + str(args.augrotate) + '_S' + str(args.seed) + '_P' + str(args.patience) + '_V' + str(args.version))
        print("="*100)

        print("="*100)
        print("train  start!")
        print("="*100)

        history = model.fit(x_train,y_train, validation_split=0.2,batch_size=args.batch_size,epochs=args.epoch,shuffle=True,callbacks=[early_cb, model_checkpoint])
        print("="*100)
        print("Done training ")
        print("="*100)

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

        # reload the model
        model.load_weights(path + '/' + model_name_bce)


    results = model.predict(x_test, verbose=1)
    results_binary = np.where(results >= threshold, 1.0, 0.0)
    saveResult(path, results_binary, loss_fun=model_loss_name + '_binaried')
    saveResult(path, results, loss_fun = model_loss_name) # True means save with _bce
    # test_loss = model.evaluate(x = x_test, y = y_true, batch_size = args.batch_size) # test loss with BCE



    # save training loss and val_loss
    SaveTestLossData(path, results, history, loss_fun=model_loss_name)



        
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


    
    return min(history.history['val_loss'])

