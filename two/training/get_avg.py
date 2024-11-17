# read the csv_record.csv file and get the rank of the model based on the metrics

import csv
import glob
import os
import numpy as np
import pandas as pd

csv_name = "csv_record"
avg_name = "avg_record"

model_num = 8
training_num = 5



# read the csv file as a two-dimensional array
with open(csv_name + ".csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data = np.array(rows)

np.set_printoptions(threshold=np.inf)

with open(avg_name + ".csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "test_bce", "test_mse", "test_mbce", "test_mmse", "test_PA", "test_mPA", "test_IOU", "test_sbce", "test_msbce", "train_sbce", "train_val_sbce", "train_msbce", "train_val_msbce", "train_PA", "train_mPA", "test_bmse", "test_mbmse", "test_bbce", "test_mbbce"])
    # get the average of the metrics
    for i in range(model_num):
        test_bce_list = []
        test_mse_list = []
        test_mbce_list = []
        test_mmse_list = []
        test_PA_list = []
        test_mPA_list = []
        test_IOU_list = []
        test_sbce_list = []
        test_msbce_list = []
        train_sbce_list = []
        train_val_sbce_list = []
        train_msbce_list = []
        train_val_msbce_list = []
        train_PA_list = []
        train_mPA_list = []
        test_bmse_list = []
        test_mbmse_list = []
        test_bbce_list = []
        test_mbbce_list = []
        for j in range(training_num):
            test_bce_list.append(float(data[i * training_num + j + 1][1]))
            test_mse_list.append(float(data[i * training_num + j + 1][2]))
            test_mbce_list.append(float(data[i * training_num + j + 1][3]))
            test_mmse_list.append(float(data[i * training_num + j + 1][4]))
            test_PA_list.append(float(data[i * training_num + j + 1][5]))
            test_mPA_list.append(float(data[i * training_num + j + 1][6]))
            test_IOU_list.append(float(data[i * training_num + j + 1][7]))
            test_sbce_list.append(float(data[i * training_num + j + 1][8]))
            test_msbce_list.append(float(data[i * training_num + j + 1][9]))
            train_sbce_list.append(float(data[i * training_num + j + 1][10]))
            train_val_sbce_list.append(float(data[i * training_num + j + 1][11]))
            train_msbce_list.append(float(data[i * training_num + j + 1][12]))
            train_val_msbce_list.append(float(data[i * training_num + j + 1][13]))
            train_PA_list.append(float(data[i * training_num + j + 1][14]))
            train_mPA_list.append(float(data[i * training_num + j + 1][15]))
            test_bmse_list.append(float(data[i * training_num + j + 1][16]))
            test_mbmse_list.append(float(data[i * training_num + j + 1][17]))
            test_bbce_list.append(float(data[i * training_num + j + 1][18]))
            test_mbbce_list.append(float(data[i * training_num + j + 1][19]))

        test_bce_avg = np.mean(test_bce_list)
        test_mse_avg = np.mean(test_mse_list)
        test_mbce_avg = np.mean(test_mbce_list)
        test_mmse_avg = np.mean(test_mmse_list)
        test_PA_avg = np.mean(test_PA_list)
        test_mPA_avg = np.mean(test_mPA_list)
        test_IOU_avg = np.mean(test_IOU_list)
        test_sbce_avg = np.mean(test_sbce_list)
        test_msbce_avg = np.mean(test_msbce_list)
        train_sbce_avg = np.mean(train_sbce_list)
        train_val_sbce_avg = np.mean(train_val_sbce_list)
        train_msbce_avg = np.mean(train_msbce_list)
        train_val_msbce_avg = np.mean(train_val_msbce_list)
        train_PA_avg = np.mean(train_PA_list)
        train_mPA_avg = np.mean(train_mPA_list)
        test_bmse_avg = np.mean(test_bmse_list)
        test_mbmse_avg = np.mean(test_mbmse_list)
        test_bbce_avg = np.mean(test_bbce_list)
        test_mbbce_avg = np.mean(test_mbbce_list)

        # write the average of the metrics to a new csv file

        
        writer.writerow([data[i * training_num + 1][0], test_bce_avg, test_mse_avg, test_mbce_avg, test_mmse_avg, test_PA_avg, test_mPA_avg, test_IOU_avg, test_sbce_avg, test_msbce_avg, train_sbce_avg, train_val_sbce_avg, train_msbce_avg, train_val_msbce_avg, train_PA_avg, train_mPA_avg, test_bmse_avg, test_mbmse_avg, test_bbce_avg, test_mbbce_avg])



print("done getting the average of the metrics!")










    




        



