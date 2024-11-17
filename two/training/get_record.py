import csv
import glob
import os


filename =  "test_result" ## need to change if the result file use different name
csv_name = "csv_record"

with open(csv_name + ".csv", 'w', newline='') as csvfile:
    dirs = glob.glob(filename + "/*")

    # sort the dirs by name
    dirs.sort()

    writer = csv.writer(csvfile)
    writer.writerow(['model_name', 'test_bce', 'test_mse', 'test_mbce', 'test_mmse', 'test_PA', 'test_mPA', 'test_IOU', 'test_sbce', 'test_msbce', 'train_bbce', 'train_val_bbce', 'train_mbbce', 'train_val_mbbce', 'train_PA', 'train_mPA', 'test_bmse', 'test_mbmse', 'test_bbce', 'test_mbbce'])
    for i in range(len(dirs)):
        path_1 = dirs[i]
        path_2 = 'record_soft_round_bce.txt'
        tmp = []
        if os.path.exists(path_1 + '/' + path_2):
            print(path_1 + '/' + path_2)
            with open(path_1 + '/' + path_2, 'r') as f:
                for line in f.read().splitlines():            
                    tmp.append(line)
                writer.writerow([dirs[i], tmp[1], tmp[3], tmp[5], tmp[7], tmp[9], tmp[11], tmp[13], tmp[15], tmp[17], tmp[19], tmp[21], tmp[23], tmp[25], tmp[27], tmp[29], tmp[31], tmp[33], tmp[35], tmp[37]])

print("get record!")


