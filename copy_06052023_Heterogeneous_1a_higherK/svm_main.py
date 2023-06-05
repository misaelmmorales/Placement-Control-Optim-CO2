import sys
sys.path.append('/home/bailianchen/er/MonitorOpt/libsvm-3.22/python')
from svmutil import *
import matk
import numpy as np

# General settings
Obj_name = 'cumulative CO2 leak'              # Objective name: not used in code, just for reference
Main_Directory = '/home/bailianchen/er/MonitorOpt'
nTrain = 80                             # The number of training simulations
Obj_filename = 'run_co2mt_his.dat'      # Objective file name
nColumn = 4                             # The column in which the object of interest is located
x_max = [-12, -18, -12, -10, 31.7]      # Upper bounds for all the uncertain parameters
x_min = [-14, -20, -14, -12, 3.17]      # Lower bounds for all the uncertain parameters


# Read data (y_train) from fehm training simulations
y_train = []
for itrain in range(1, nTrain+1):
    train_filename = Main_Directory + '/workdir.' + str(itrain) + '/' + Obj_filename
    with open(train_filename) as f1:
        lines1 = f1.readlines()
        lines1 = [line.rstrip('\n') for line in open(train_filename)]
        lastlines = lines1[-1]
        Objective = float(lastlines.split()[nColumn-1])
        y_train.append(Objective)

# Read data (x_train) from 'sample.txt' file
#x_train = []
#count=0
#for lines2 in open('sample.txt'):
#    count += 1
#    lines2 = lines2.rstrip('\n')
#    lines2 = lines2.split(None, 1)
#    label, features = lines2
#    xi={}
#    dim_count = 0
#    if count>3:  # The first three lines in 'sample.txt' will not be recorded  
#        for e in features.split():
#            dim_count += 1
#            val = e   
#            xi[int(dim_count)] = float(val)
#        x_train += [xi]

m = matk.matk()
ss = m.read_sampleset('sample.txt')
x_train = ss.samples.values

# Scale x_train to [-1,1]
x_train_scaled = np.zeros_like(x_train)
for i in range(0,100):
    for j in range(0,5):
        x_train_scaled[i][j] = 2*(x_train[i][j] - x_min[j])/(x_max[j] - x_min[j]) - 1

              
# Read data in LIBSVM format (Option 2)
#y_train, x_train = svm_read_problem('/home/bailianchen/er/MonitorOpt/Training_DataSet.txt')

# Train the LHS samples and construct the SVM ROMs
m = svm_train(y_train[:80], x_train_scaled[:80].tolist(), '-s 4')

# Predict the response of interest using SVM ROMS
p_label, p_acc, p_val = svm_predict([0]*len(x_train_scaled), x_train_scaled, m)

