import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from pyearth import Earth
from matk import matk, pest_io
from Uncertainty import *
from uncertaintyMetric import *


# General settings
Total_time       = 1800                        # Total injection and post-injection time (days)
Obj_name         = 'cumulative CO2 leak'       # Objective name: not used in code, just for reference
Main_Directory   = os.getcwd()                 # '/scratch/sft/bailian_chen/NRAP/SyntheticCase/Heterogeneous_1a_higherK'
Data_Directory   = os.path.join(Main_Directory, 'data')

nTrain           = 500                         # The number of training simulations
Obj_filename     = 'run_co2mt.his'             # Objective file name
NLeakPotential   = 3                           # Number of potential leak point 
nColumn_obj      = 50                          # The column(s) in which the object of interest is located
MeasureType      = 4                           # Measurement type: 1 for pressure; 2 for CO2 saturation; 3 for tempeture, 4 for pressure+CO2 saturation                     
nColumn_data     = [7]                         # The column in which the data measurement is located 
nTimeSeries      = 60                          # The number of measurement time series
nInterval        = 1                           # Measurement interval, default value = 1/Month

x_max            = [1e-14, 1e-14, 1e-14, 2.0]  # Upper bounds for all the uncertain parameters
x_min            = [1e-19, 1e-19, 1e-19, 0.5]  # Lower bounds for all the uncertain parameters
nMCSamples       = 100000                      # Number of monte carlo samples
nParam           = 4                           # Number of uncertain parameters
nDataRealization = 200                         # Number of data realizations
err_option       = 3                           # Type of err option
time_sensitivity = 1                           # Whether consider time sensitivity for UR
post_processing  = 1                           # For ploting
ROMs_validation  = 0                           # ROMs cross-validation 
print('Current Working Directory: {}'.format(Main_Directory))
print('Data Directory: {}'.format(Data_Directory))

## Step 1: Perfrom training simulation
# This step is done in a seperate code.

## Step 2: Read-in training simulation results
# Read data (data_train) from fehm training simulations
titles = ['.', 'Pressure', 'CO2 Saturation (l)', 'Temperature', 'Pressure + CO2 Saturation']
if MeasureType == 1:    # for pressure
    data_filename = 'run_presWAT.his'            # monitoring data file name
    eps = 0.002                                  # History match error tolerance
elif MeasureType == 2:  # for liquid CO2 saturation
    data_filename = 'run_co2sl.his'
    eps =  0.05 
elif MeasureType == 3:  # for tempeture
    data_filename = 'run_temp.his' 
    eps = 0.002 
elif MeasureType == 4:  # for pressure+CO2 saturation
    data_filename = ['run_presWAT.his','run_co2sl.his']
    eps = [0.002, 0.05]
else:
    print("No such measurement option, optimization will be terminated")
print('Measure Type: {} | Data File Name: {} | Description: {} | HM-epsilon: {}'.format(MeasureType, data_filename, titles[MeasureType], eps))

nData_read =  nTimeSeries*len(nColumn_data)*nInterval  # The actural number of data read from simulation output
nData      =  nTimeSeries*len(nColumn_data)            # The number of measurement data points

data_train_read_raw  = np.zeros((nTrain,nData_read))
data_train_read_raw0 = np.zeros((nTrain,len(nColumn_data)))
data_train_read      = np.zeros((nData_read,nTrain))
data_train           = np.zeros((nData,nTrain))
time_set             = []
time_point_read      = np.zeros(nTimeSeries*nInterval+1)
time_point           = np.zeros(nTimeSeries+1)

for i in range(0, len(nColumn_data)):
    for itrain in range(0, nTrain):
        if MeasureType==4:
            #train_data_filename = Main_Directory + '/workdir.' + str(itrain+1) + '/' + data_filename[i]
            train_data_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), data_filename[i])
        else:
            #train_data_filename = Main_Directory + '/workdir.' + str(itrain+1) + '/' + data_filename
            train_data_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), data_filename)
        count = 0
        for lines in open(train_data_filename):
            count += 1
            lines = lines.rstrip('\n')
            lines = lines.split(None, 1)
            times, features = lines
            dim_count = 0
            if count==5:
                for e in features.split():
                    dim_count += 1
                    val = e
                    if dim_count == nColumn_data[i]-1:
                        xi = float(val)
                data_train_read_raw0[itrain][i] = xi
            if count>5:
                for e in features.split():
                    dim_count += 1
                    val = e
                    if dim_count == nColumn_data[i]-1:
                        xi = float(val)                        
                time_set += [float(times)]
                data_train_read_raw[itrain][count-6+i*nTimeSeries*nInterval] = xi

time_point_read[1:nTimeSeries*nInterval+1] = time_set[:nTimeSeries*nInterval]
data_train_read = data_train_read_raw.T

if nData_read==nData:
    data_train = data_train_read
    time_point = time_point_read
else:
    multiplier=nData_read/nData
    for i in range(0,nData):
        data_train[i] = data_train_read[multiplier*i+1]
        if i<=nTimeSeries:
            time_point[i] = time_point_read[multiplier*i]
    time_point[nTimeSeries] = Total_time
print('Read data from fehm traning simulations: Done!')
print('Data Train: {} | Time Point: {}'.format(data_train.shape, time_point.shape))

# Read response of interest (y_train) from fehm training simulations
y_train = np.zeros(nTrain)
for itrain in range(0, nTrain):
    #train_filename = Main_Directory + '/workdir.' + str(itrain+1) + '/' + Obj_filename
    train_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), Obj_filename)
    with open(train_filename) as f1:
        lines1 = f1.readlines()
        lines1 = [line.rstrip('\n') for line in open(train_filename)]
        lastlines = lines1[-1]
        Objective = float(lastlines.split()[nColumn_obj-1])
        y_train[itrain] = Objective
print('Read y_train from fehm traning simulations: Done!')
print('y_train: {}'.format(y_train.shape))

# Read data (x_train) from 'sample.txt' file
m       = matk()
ss      = m.read_sampleset('sample.txt')
x_train = ss.samples.values
print('Read x_train from fehm traning simulations: Done!')
print('x_train: {}'.format(x_train.shape))

# Scale x_train to [-1,1]
x_train_scaled = np.zeros((nTrain,nParam))
for i in range(0,nTrain):
    for j in range(0,nParam):
        x_train_scaled[i][j] = 2*(x_train[i][j] - x_min[j])/(x_max[j] - x_min[j]) - 1
print('Rescale x_train to [-1,1]: Done!')

## Step 3: 10-fold cross-validation of ROMs
if ROMs_validation==1:
    print('ROMs accuracy validation: 10-fold cross-validation')
    # ROMs validation for objs
    Interval     = int(nTrain/10)
    predict_obj  = np.zeros(nTrain)
    predict_data = np.zeros((nData,nTrain))
    
    for i in range(0,10):
        x_train_scaled_v = x_train_scaled.tolist()
        y_train_v        = y_train.tolist()
        data_train_raw_v = data_train.T.tolist()
        
        del x_train_scaled_v[Interval*i:Interval*(i+1)]
        del y_train_v[Interval*i:Interval*(i+1)]
        del data_train_raw_v[Interval*i:Interval*(i+1)]
        
        x_train_scaled_v = np.array(x_train_scaled_v)
        data_train_v     = np.array(data_train_raw_v).T
        data_train_v1    = data_train_v
        
        # build ROMs
        ROM_obj = Earth() #(max_degree=2)        
        ROM_obj.fit(x_train_scaled_v,y_train_v)
        ROM_data = {}
        for iData in range(0,nData):
            ROM_data[iData] = Earth() #(max_degree=2)
            ROM_data[iData].fit(x_train_scaled_v,data_train_v1[iData])            
        # predict   
        predict_obj[Interval*i:Interval*(i+1)] = ROM_obj.predict(x_train_scaled[Interval*i:Interval*(i+1)])
        for iData in range(0,nData):
            predict_data[iData][Interval*i:Interval*(i+1)] = ROM_data[iData].predict(x_train_scaled[Interval*i:Interval*(i+1)])
    
    for i in range(0,nData):
        for j in range(0,nTrain):
            if predict_data[i][j]<np.amin(data_train_read_raw):
                predict_data[i][j] = np.amin(data_train_read_raw)
            if predict_data[i][j]>np.amax(data_train_read_raw):
                predict_data[i][j] = np.amax(data_train_read_raw)
        
    for i in range(0,nTrain):
        if predict_obj[i]<0:
            predict_obj[i] = 0
    CorreCoeff = np.corrcoef(predict_obj,y_train)[0][1]
    print('The correlation coefficient between the true values and the predicted values for Obj ROMs: '+ str(CorreCoeff))
    
    # plot1
    CorreCoef_data = np.zeros(nData)
    for i in range (0,nData):        
        CorreCoeff2 = np.corrcoef(predict_data[i],data_train[i])[0][1]        
        print('The correlation coefficient between the true values and the predicted values for ROM data point '+str(i+1)+': '+ str(CorreCoeff2))
        CorreCoef_data[i] = CorreCoeff2
        
        plt.figure()
        plt.scatter(predict_data[i],data_train[i],marker='*',color='blue')
        real_max0, pred_max0 = max(data_train[i]),       max(predict_data[i])
        maxV0,     minV0     = max(real_max0,pred_max0), min(real_min0,pred_min0)
        real_min0, pred_min0 = min(data_train[i]),       min(predict_data[i])
        plt.plot([minV0,maxV0],[minV0,maxV0],ls="-",c="0.3")
        plt.xlabel('ROM Prediction (MPa)',fontsize=16,fontweight="bold"); plt.ylabel('True Value from Simulation (MPa)',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
        plt.xlim([minV0,maxV0]); plt.ylim([minV0,maxV0])
        figname = 'figures/ROMsData-validation'+str(i+1); plt.savefig(figname,bbox_inches='tight') #; plt.show()
        plt.close()
        
    plt.figure()
    plt.scatter(np.arange(1,nData+1,1),CorreCoef_data,marker='d',color='red')
    plt.xlabel('Monitoring Data Point',fontsize=16,fontweight="bold"); plt.ylabel('Correlation Coefficient',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
    plt.xlim([0,nData+1]); plt.ylim([0.9,1])
    plt.savefig("figures/CorrelationCoeff_data",bbox_inches='tight'); plt.show()
    
    # plot2
    plt.figure()
    plt.scatter(predict_obj/1e6,y_train/1e6,marker='*',color='blue')
    real_max1, pred_max1 = max(y_train/1e6), max(predict_obj/1e6)
    maxV1 = max(real_max1,pred_max1)+5
    plt.plot([0,maxV1],[0,maxV1],ls="-",c="0.3")
    plt.xlabel('ROM Prediction (Kt)',fontsize=16,fontweight="bold"); plt.ylabel('True Value from Simulation (Kt)',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
    plt.xlim([0,maxV1]); plt.ylim([0,maxV1])
    plt.savefig("rigures/ROMs-obj",bbox_inches='tight'); plt.show()

        
## Step 4: Construct the Mars ROMs for data and response of interest
# ROMs for data points
ROM_data = {}
for iData in range(0, nData):
    ROM_data[iData] = Earth()#(max_degree=2)
    ROM_data[iData].fit(x_train_scaled[:nTrain], data_train[iData])
print('Build the ROMs for data points: Done!')

# ROMs for obj
ROM_obj = Earth()#(max_degree=2)
ROM_obj.fit(x_train_scaled[:nTrain],y_train)
print('Build the ROMs for objective of interests: Done!')


## Step 5: Generate Monte Carlo(MC) samples
np.random.seed(787878)
mc_design = np.random.rand(nMCSamples, nParam)
mc_design = mc_design*2 - 1
print('Generate Monte Carlo samples: Done!')


### Step 6: Evaluate the MC samples using the built ROMs for data points/objs
print('Evaluating Monte Carlo samples: ing... ing ...')
mc_data = np.zeros((nData, nMCSamples))
for iData in range(0, nData):
    mc_data[iData] = ROM_data[iData].predict(mc_design)
    for iMCSamples in range(0,nMCSamples):
        if mc_data[iData][iMCSamples]<np.amin(data_train_read_raw):
            mc_data[iData][iMCSamples] = np.amin(data_train_read_raw)
        if mc_data[iData][iMCSamples]>np.amax(data_train_read_raw):
            mc_data[iData][iMCSamples] = np.amax(data_train_read_raw)

mc_obj = np.zeros(nMCSamples)   
mc_obj = ROM_obj.predict(mc_design)
for i in range(0,nMCSamples):
    if mc_obj[i]<0:
        mc_obj[i] = 0
print('Evaluate the Monte Carlo samples: Done!')

## Step 7: Calculate posterior distribution and uncertainty reduction
prior_mean    = sum(mc_obj)/len(mc_obj)
prior_p90mp10 = uncertaintyMetric(mc_obj)

# Generate synthetic data
#synthetic_data = mc_data.T[:nDataRealization]
def fehm(p):
    print('')
p = matk(model=fehm)
p.add_par('perm4',min=x_min[0], max=x_max[0])
p.add_par('perm5',min=x_min[1], max=x_max[1])
p.add_par('perm6',min=x_min[2], max=x_max[2])
p.add_par('kmult',min=x_min[3], max=x_max[3])
s = p.lhs(siz=nDataRealization, seed=1000)

LHSamples        = s.samples.values
LHSamples_scaled = np.zeros_like(LHSamples)

for i in range(0,nDataRealization):
    for j in range(0,nParam):
        LHSamples_scaled[i][j] = 2*(LHSamples[i][j] - x_min[j]) / (x_max[j] - x_min[j]) - 1

synthetic_data_raw = np.zeros((nData, nDataRealization))
for iData in range(0, nData):
    synthetic_data_raw[iData] = ROM_data[iData].predict(LHSamples_scaled)

synthetic_data = synthetic_data_raw.T
for i in range(0,nDataRealization):
    for j in range(0,nData):
        if synthetic_data[i][j]<np.amin(data_train_read_raw):
            synthetic_data[i][j] = np.amin(data_train_read_raw)
        if synthetic_data[i][j]>np.amax(data_train_read_raw):
            synthetic_data[i][j] = np.amax(data_train_read_raw)
print('Generate synthetic monitoring data: Done!')

# Calculate posterior metrics
post_p90mp10_mean, post_p90mp10_time, post_p90mp10_iData, post_mean, post_mean_iData, nSamples,mc_obj_post = Uncertainty_calc(mc_data, synthetic_data, mc_obj, err_option, eps, MeasureType, time_sensitivity, len(nColumn_data))
print('\nData assimilation is done!\n')

## Step 8: Post-processing
if post_processing==1:
    print('Post-processing')
    
    # plot posterior uncertainty change with time
    plt.figure()
    U = np.insert(post_p90mp10_time/1e6,0,prior_p90mp10/1e6)
    #T = np.insert(time_point,0,0)
    plt.plot(np.array(time_point)/360,U,marker='o',markersize=4,markerfacecolor='none',color='red',label='M1')
    #plt.plot(np.array(time_point)/360,U_2,marker='^',markersize=4,markerfacecolor='none',color='blue',label='M2')
    #plt.plot(np.array(time_point)/360,U_4,marker='x',markersize=4,color='black',label='M1+M2')
    #plt.plot(np.array(time_point)/360,U_3,marker='+',markersize=5,color='green',label='M1+M2+M3')    
    plt.xlabel('Time (Years)',fontsize=16,fontweight="bold")
    plt.ylabel('U of Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.xlim([0,Total_time/360])
    plt.ylim([0,40])
    plt.legend(loc='upper right',fontsize=14)
    #plt.savefig("figures/UR_ab.png",bbox_inches='tight')
    plt.show()
    print('The prior uncertainty: ' + str(U[0]))
    print('The end point posterior uncertainty: ' + str(U[len(U)-1]))
    
    # plot U values of prior and posterior distributions
    plt.figure()
    plt.scatter(1,prior_p90mp10/1e6,marker='d',color='red')
    xl=np.zeros(nDataRealization)
    for i in range(0,nDataRealization):
        xl[i]= 2
    plt.scatter(xl,post_p90mp10_iData/1e6,marker='d',color='blue')
    plt.plot((0.7,2.3),(post_p90mp10_mean/1e6,post_p90mp10_mean/1e6),ls='--',color='blue')
    #plt.plot((1.1,1.1),(post_p90mp10_mean,prior_p90mp10),ls='-',color='red')
    plt.annotate('',xy=(1.1,post_p90mp10_mean/1e6),xycoords='data',xytext=(1.1,prior_p90mp10/1e6),textcoords='data',arrowprops={'arrowstyle':'<->','color':'red','lw':'1.5'})
    plt.figtext(0.45,0.7,"Uncertainty Reduction",fontsize=14, fontweight='bold', color='red')
    #plt.xlabel('',fontsize=14)
    plt.ylabel('U of Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
    plt.xticks(np.arange(4),('','Prior','Posterior',''),fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.ylim([0,40])
    #plt.legend(loc='center right')
    plt.savefig("figures/U of prior and posterior",bbox_inches='tight')
    plt.show()
    
    # plot CDF for prior and posterior distribution of obj
    plt.figure()
    num_bins =100
    hist, bin_edges = np.histogram(mc_obj/1e6, bins=num_bins)
    cdf = np.cumsum(hist)
    cdf1=np.zeros(num_bins)
    for i in range(0,num_bins):
        cdf1[i]=float(cdf[i])/nMCSamples
    plt.plot(bin_edges[1:],cdf1,'r',label="Prior")
    for i in range(0,len(mc_obj_post)):
        hist_post, bin_edges_post = np.histogram(mc_obj_post[i]/1e6, bins=num_bins)
        cdf_post = np.cumsum(hist_post)
        cdf1_post=np.zeros(num_bins)
        for j in range(0,num_bins):
            cdf1_post[j]=float(cdf_post[j])/len(mc_obj_post[i])
        if i==0:
            plt.plot(bin_edges_post[1:],cdf1_post,'b',label="Posterior")
        else:
            plt.plot(bin_edges_post[1:],cdf1_post,'b')
    plt.xlabel('Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
    plt.ylabel('CDF',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.legend(loc='upper left',fontsize=14)
    plt.savefig("figures/CDF.png",bbox_inches='tight')    
    plt.show()

    #plot histogram
    plt.figure()
    num_bins1=50
    plt.hist(mc_obj/1e6,bins=num_bins1,color='blue',label='Prior')
    plt.hist(mc_obj_post[0]/1e6,bins=num_bins1,color='orange',label='Posterior_R1')
    plt.hist(mc_obj_post[99]/1e6,bins=num_bins1,color='red',label='Posterior_R100')
    plt.xlabel('Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")   
    plt.ylabel('Frequency',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.legend(loc='upper left')
    plt.savefig("figures/histogram.png",bbox_inches='tight')
    plt.show()
    
    # plot the data realization
    for iLoc in range(0, len(nColumn_data)):
        plt.figure()
        for i in range(0,nDataRealization):
            data_plot=np.insert(synthetic_data[i][nTimeSeries*iLoc:nTimeSeries*(iLoc+1)],0,data_train_read_raw0[0][iLoc])
            plt.plot(np.array(time_point)/360,data_plot)
        plt.xlabel('Time (Years)',fontsize=16,fontweight="bold")
        plt.ylabel('Monitoring Pressure (MPa)',fontsize=16,fontweight="bold")
        #plt.ylabel('CO$_2$ Saturation',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.xlim([0,Total_time/360])
        #plt.ylim([10.200,10.235])
        figname = 'figures/data_realizations_Loc_'+str(iLoc+1)
        plt.savefig(figname,bbox_inches='tight')
        plt.show()
    
    # plot number of samples remained
    plt.figure()
    plt.scatter(np.arange(1,nDataRealization+1,1),nSamples,marker='o',color='blue')
    plt.xlabel('Data Realization',fontsize=16,fontweight="bold")
    plt.ylabel('Samples Remained',fontsize=16,fontweight="bold")
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.xlim([0,nDataRealization])
    #plt.ylim([0,nMCSamples])
    plt.savefig("figures/samples_remained.png",bbox_inches='tight')
    plt.show()
    
    # plot boxplot for the number of samples remained
    #plt.figure()
    #data_box=[nSamples_1,nSamples]
    #plt.boxplot(data_box, 0, '')
    #plt.ylabel('Samples Remained',fontsize=16,fontweight="bold")
    #plt.rc('ytick',labelsize=14)
    #plt.xticks([1, 2], ['M1', 'M2'],fontsize=16,fontweight="bold")
    #plt.savefig("figures/boxplot_samples_remained.png",bbox_inches='tight')
    #plt.show()
    
    print('\nWorkflow: done successfully!\n\n')