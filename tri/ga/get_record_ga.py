import csv
import glob
import os

dir_result = "test_result"## need to change
filename =  dir_result 
csv_name = "csv_record"

with open(csv_name + ".csv", 'w', newline='') as csvfile:
    dirs = glob.glob(filename + "/*")
    writer = csv.writer(csvfile)
    writer.writerow([' ','test_loss_bce', 'test_loss_mse', 'test_my_loss_bce', 'test_my_loss_mse', 'PA', 'modified_PA', 'val_my_loss'])
    for i in range(len(dirs)):
        path_1 = dirs[i]
        path_2 = 'record_binary_bce.txt'
        tmp = []
        if os.path.exists(path_1 + '/' + path_2):
            with open(path_1 + '/' + path_2, 'r') as f:
                for line in f.read().splitlines():            
                    tmp.append(line)
                writer.writerow([dirs[i], tmp[1], tmp[3], tmp[5], tmp[7], tmp[9] , tmp[11] , tmp[13]])

print("get record!")

