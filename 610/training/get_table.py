import csv
import numpy as np


filename = "csv_record.csv"
tablename = 'table_record.csv'
training_times = 5
model_num = 8
BCE_column = 1
MSE_column = 2
mBCE_column = 3
mMSE_column = 4
PA_column = 5
MPA_column = 6
TEST_SBCE_column = 8
TEST_MSBCE_column = 9
TRAIN_SBCE_column = 10
VAL_SBCE_column = 11
TEST_BBCE_column = 18
TEST_MBBCE_column = 19



with open(filename, 'r', newline='') as file:
    # create csv reader
    csv_reader = csv.reader(file)

    data = np.array(list(csv_reader))
    print(data.shape)
    file.close()


with open(tablename,'w', newline='') as file:
    # create csv writer
    csv_writer = csv.writer(file)

    # Write the header row
    csv_writer.writerow(['Name', 'TRAIN_SBCE', 'VAL_SBCE', 'TEST_SBCE', 'TEST_MSBCE', 'TEST_PA', 'TEST_MPA'])
    for i in range(model_num):
        if i == 7:
            i = -1 # base 

        name = data[1+(i+1)*training_times, 0] # 0 is name
        TRAIN_SBCE_list = []
        VAL_SBCE_list = []
        SBCE_list = []
        MSBCE_list = []
        PA_list = []
        MPA_list = []
        for j in range(training_times):
            TRAIN_SBCE_list.append(float(data[1+(i+1)*training_times+j, TRAIN_SBCE_column])) 
            VAL_SBCE_list.append(float(data[1+(i+1)*training_times+j, VAL_SBCE_column])) 
            SBCE_list.append(float(data[1+(i+1)*training_times+j, TEST_SBCE_column]))
            MSBCE_list.append(float(data[1+(i+1)*training_times+j, TEST_MSBCE_column])) 
            PA_list.append(float(data[1+(i+1)*training_times+j, PA_column])) 
            MPA_list.append(float(data[1+(i+1)*training_times+j, MPA_column]))


        csv_writer.writerow([name, np.mean(TRAIN_SBCE_list), np.mean(VAL_SBCE_list), np.mean(SBCE_list), np.mean(MSBCE_list), np.mean(PA_list)*100, np.mean(MPA_list)*100])
        csv_writer.writerow(["Min", np.min(TRAIN_SBCE_list), np.min(VAL_SBCE_list), np.min(SBCE_list), np.min(MSBCE_list), np.min(PA_list)*100, np.min(MPA_list)*100])
        csv_writer.writerow(["Max", np.max(TRAIN_SBCE_list), np.max(VAL_SBCE_list), np.max(SBCE_list), np.max(MSBCE_list), np.max(PA_list)*100, np.max(MPA_list)*100])


        if i == -1:
            break
print("done ")









    