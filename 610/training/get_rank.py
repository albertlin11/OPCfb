# derive the rank from the metrics of data we read
import csv
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata


avg_name = "avg_record"
rank_name = "rank_record"

# read the csv file as a two-dimensional array
with open(avg_name + ".csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
data = np.array(rows)



np.set_printoptions(threshold=np.inf)

#PA is negative because we want to rank it in descending order
test_bce_list = [float(value) for value in data[1:, 1]]
test_mse_list = [float(value) for value in data[1:, 2]]
test_mbce_list = [float(value) for value in data[1:, 3]]
test_mmse_list = [float(value) for value in data[1:, 4]]
test_PA_list = [-float(value) for value in data[1:, 5]]
test_mPA_list = [-float(value) for value in data[1:, 6]]
test_IOU_list = [float(value) for value in data[1:, 7]]
test_sbce_list = [float(value) for value in data[1:, 8]]
test_msbce_list = [float(value) for value in data[1:, 9]]
train_sbce_list = [float(value) for value in data[1:, 10]]
train_val_sbce_list = [float(value) for value in data[1:, 11]]
train_msbce_list = [float(value) for value in data[1:, 12]]
train_val_msbce_list = [float(value) for value in data[1:, 13]]
train_PA_list = [-float(value) for value in data[1:, 14]]
train_mPA_list = [-float(value) for value in data[1:, 15]]
test_bmse_list = [float(value) for value in data[1:, 16]]
test_mbmse_list = [float(value) for value in data[1:, 17]]
test_bbce_list = [float(value) for value in data[1:, 18]]
test_mbbce_list = [float(value) for value in data[1:, 19]]

# get the rank of the model based on the metrics 
test_bce_rank = [int(i) for i in rankdata(test_bce_list)]
test_mse_rank = [int(i) for i in rankdata(test_mse_list)]
test_mbce_rank = [int(i) for i in rankdata(test_mbce_list)]
test_mmse_rank = [int(i) for i in rankdata(test_mmse_list)]
test_PA_rank = [int(i) for i in rankdata(test_PA_list)]
test_mPA_rank = [int(i) for i in rankdata(test_mPA_list)]
test_IOU_rank = [int(i) for i in rankdata(test_IOU_list)]
test_sbce_rank = [int(i) for i in rankdata(test_sbce_list)]
test_msbce_rank = [int(i) for i in rankdata(test_msbce_list)]
train_sbce_rank = [int(i) for i in rankdata(train_sbce_list)]
train_val_sbce_rank = [int(i) for i in rankdata(train_val_sbce_list)]
train_msbce_rank = [int(i) for i in rankdata(train_msbce_list)]
train_val_msbce_rank = [int(i) for i in rankdata(train_val_msbce_list)]
train_PA_rank = [int(i) for i in rankdata(train_PA_list)]
train_mPA_rank = [int(i) for i in rankdata(train_mPA_list)]
test_bmse_rank = [int(i) for i in rankdata(test_bmse_list)]
test_mbmse_rank = [int(i) for i in rankdata(test_mbmse_list)]
test_bbce_rank = [int(i) for i in rankdata(test_bbce_list)]
test_mbbce_rank = [int(i) for i in rankdata(test_mbbce_list)]


# get the rank of the model based on the metrics
with open(rank_name + ".csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "test_bce", "test_mse", "test_mbce", "test_mmse", "test_PA", "test_mPA", "test_IOU", "test_sbce", "test_msbce", "train_sbce", "train_val_sbce", "train_msbce", "train_val_msbce", "train_PA", "train_mPA", "test_bmse", "test_mbmse", "test_bbce", "test_mbbce"])
    # get the rank of the model based on the metrics

    # write the rank of the model based on the metrics to a new csv file
    for i in range(len(data) - 1):
        writer.writerow([data[i + 1][0], test_bce_rank[i], test_mse_rank[i], test_mbce_rank[i], test_mmse_rank[i], test_PA_rank[i], test_mPA_rank[i], test_IOU_rank[i], test_sbce_rank[i], test_msbce_rank[i], train_sbce_rank[i], train_val_sbce_rank[i], train_msbce_rank[i], train_val_msbce_rank[i], train_PA_rank[i], train_mPA_rank[i], test_bmse_rank[i], test_mbmse_rank[i], test_bbce_rank[i], test_mbbce_rank[i]])


print("done writing rank")