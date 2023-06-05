import sys
import numpy as np
from pyearth import Earth
from Uncertainty import *
from uncertaintyMetric import *
from matk import matk, pest_io
from matplotlib import pyplot as plt


# General settings
Total_time=1800                               # Total injection and post-injection time (days)
Obj_name = 'cumulative CO2 leak'             # Objective name: not used in code, just for reference
Main_Directory = '/scratch/sft/bailian_chen/Heterogeneous_1a'
nTrain = 500                                 # The number of training simulations
Obj_filename = 'run_co2mt.his'               # Objective file name
NLeakPotential = 3                           # Number of potential leak point 
nColumn_obj = [4,6,13]                       # The column(s) in which the object of interest is located
#data_filename = 'run_presWAT.his'            # monitoring data file name
#data_filename = 'run_temp.his' 
data_filename = 'run_co2sl.his'
#eps = 0.001  # for pressure                  # History match error tolerance
#eps = 0.001 # for tempeture
eps =  0.05 # for liquid CO2 saturation

nColumn_data = 25                             # The column in which the data measurement is located 
nData =  60                                  # The number of data points
x_max = [1e-14, 1e-14, 1e-14, 2.0]           # Upper bounds for all the uncertain parameters
x_min = [1e-19, 1e-19, 1e-19, 0.5]           # Lower bounds for all the uncertain parameters
nMCSamples = 100000                          # Number of monte carlo samples
nParam = 4                                   # Number of uncertain parameters
nDataRealization = 200                       # Number of data realizations
err_option = 3                               # Type of err option
time_sensitivity = 1                         # Whether consider time sensitivity for UR
post_processing = 1                          # For ploting
ROMs_validation = 0                          # ROMs cross-validation 



## Step 1: Perfrom training simulation
# This step is done in a seperate code.
  

## Step 2: Read-in training simulation results
# Read data (data_train) from fehm training simulations
data_train_raw = np.zeros((nTrain,nData+1))
data_train     = np.zeros((nData,nTrain))
time_set = []
for itrain in range(0, nTrain):
    train_data_filename = Main_Directory + '/workdir.' + str(itrain+1) + '/' + data_filename
    count = 0
    for lines in open(train_data_filename):
        count += 1
        lines = lines.rstrip('\n')
        lines = lines.split(None, 1)
        times, features = lines
        dim_count = 0
        if count>4:
            for e in features.split():
                dim_count += 1
                val = e
                if dim_count == nColumn_data-1:
                    xi = float(val)
            time_set   += [float(times)]
            data_train_raw[itrain][count-5] = xi
time_point = time_set[:nData+1]
data_train = data_train_raw.T[1:nData+1]
print 'Read data from fehm traning simulations: Done!'

# Read response of interest (y_train) from fehm training simulations
y_train_raw = np.zeros((nTrain,NLeakPotential))
y_train     = np.zeros((NLeakPotential,nTrain))
for itrain in range(0, nTrain):
    train_filename = Main_Directory + '/workdir.' + str(itrain+1) + '/' + Obj_filename
    with open(train_filename) as f1:
        lines1 = f1.readlines()
        lines1 = [line.rstrip('\n') for line in open(train_filename)]
        lastlines = lines1[-1]
        for iLeak in range(0,NLeakPotential):
            Objective = float(lastlines.split()[nColumn_obj[iLeak]-1])
            y_train_raw[itrain][iLeak] = Objective
y_train = y_train_raw.T
print 'Read y_train from fehm traning simulations: Done!'

# Read data (x_train) from 'sample.txt' file
m = matk()
ss = m.read_sampleset('sample.txt')
x_train = ss.samples.values
print 'Read x_train from fehm traning simulations: Done!'

# Scale x_train to [-1,1]
x_train_scaled = np.zeros_like(x_train)
for i in range(0,nTrain):
    for j in range(0,nParam):
        x_train_scaled[i][j] = 2*(x_train[i][j] - x_min[j])/(x_max[j] - x_min[j]) - 1


## Step 3: 10-fold cross-validation of ROMs
if ROMs_validation==1:
    print 'ROMs accuracy validation: 10-fold cross-validation'
    # ROMs validation for objs
    Interval = int(nTrain/10)
    predict_obj = np.zeros((NLeakPotential,nTrain ))
    predict_data = np.zeros((nData,nTrain))
    for i in range(0,10):
        x_train_scaled_v=x_train_scaled.tolist()
        y_train_raw_v=y_train_raw.tolist()
        data_train_raw_v=data_train_raw.tolist()
        del x_train_scaled_v[Interval*i:Interval*(i+1)]
        del y_train_raw_v[Interval*i:Interval*(i+1)]
        del data_train_raw_v[Interval*i:Interval*(i+1)]
        x_train_scaled_v=np.array(x_train_scaled_v)
        y_train_v=np.array(y_train_raw_v).T
        data_train_v=np.array(data_train_raw_v).T
        data_train_v1=data_train_v[1:nData+1]
        # build ROMs
        ROM_obj = {}
        ROM_data = {}
        for iLeak in range(0,NLeakPotential):
            ROM_obj[iLeak] = Earth()
            ROM_obj[iLeak].fit(x_train_scaled_v,y_train_v[iLeak])
        for iData in range(0,nData):
            ROM_data[iData] = Earth()
            ROM_data[iData].fit(x_train_scaled_v,data_train_v1[iData])            
        # predict   
        for iLeak in range(0,NLeakPotential):
            predict_obj[iLeak][Interval*i:Interval*(i+1)] = ROM_obj[iLeak].predict(x_train_scaled[Interval*i:Interval*(i+1)])
        for iData in range(0,nData):
            predict_data[iData][Interval*i:Interval*(i+1)] = ROM_data[iData].predict(x_train_scaled[Interval*i:Interval*(i+1)])
            
    for i in range(0,NLeakPotential):
        for j in range(0,nTrain):
            if predict_obj[i][j]<0:
                predict_obj[i][j]=0
    predict_obj_sum = sum(predict_obj,0)
    y_train_sum = sum(y_train,0)
    for i in range(0,nTrain):
        if predict_obj_sum[i]<0:
            predict_obj_sum[i]=0
    CorreCoeff=np.corrcoef(predict_obj_sum,y_train_sum)[0][1]
    print 'The correlation coefficient between the true values and the predicted values for Obj_sum ROMs: '+ str(CorreCoeff)

    # plot1
    CorreCoef_obj = np.zeros(NLeakPotential+1)
    for i in range (0,NLeakPotential):
        CorreCoeff1=np.corrcoef(predict_obj[i],y_train[i])[0][1]        
        print 'The correlation coefficient between the true values and the predicted values for ROM obj '+str(i+1)+': '+ str(CorreCoeff1)
        CorreCoef_obj[i]=CorreCoeff1
        plt.figure()
        plt.scatter(predict_obj[i],y_train[i],marker='*',color='blue')
        real_max=max(y_train[i])
        pred_max=max(predict_obj[i])
        maxV=max(real_max,pred_max)+500
        plt.plot([0,maxV],[0,maxV],ls="-",c="0.3")
        plt.xlabel('ROM Prediction (Kg)',fontsize=14)
        plt.ylabel('True Value from Simulation (Kg)',fontsize=14)
        plt.xlim([0,maxV])
        plt.ylim([0,maxV])
        figname = 'ROMsObj-validation'+str(i+1)
        plt.savefig(figname,bbox_inches='tight')
        #plt.show()
        plt.close()
    CorreCoef_obj[NLeakPotential]=CorreCoeff  #for obj_sum
    plt.figure()
    plt.scatter(np.arange(1,NLeakPotential+2,1),CorreCoef_obj,marker='d',color='red')
    plt.xlabel('Objective of Interest',fontsize=14)
    plt.ylabel('Correlation Coefficient',fontsize=14)
    #plt.xlim([0,NLeakPotential])
    plt.ylim([0.95,1])
    plt.xticks(np.arange(6),('','Leak #1','Leak #2','Leak #3','Total',''))
    plt.savefig("CorrelationCoeff_data",bbox_inches='tight')
    plt.show()
    
    # plot2
    CorreCoef_data = np.zeros(nData)
    for i in range (0,nData):        
        CorreCoeff2=np.corrcoef(predict_data[i],data_train[i])[0][1]        
        print 'The correlation coefficient between the true values and the predicted values for ROM data point '+str(i+1)+': '+ str(CorreCoeff2)
        CorreCoef_data[i]=CorreCoeff2
        plt.figure()
        plt.scatter(predict_data[i],data_train[i],marker='*',color='blue')
        real_max0=max(data_train[i])
        pred_max0=max(predict_data[i])
        maxV0=max(real_max0,pred_max0)
        real_min0=min(data_train[i])
        pred_min0=min(predict_data[i])
        minV0=min(real_min0,pred_min0)
        plt.plot([minV0,maxV0],[minV0,maxV0],ls="-",c="0.3")
        plt.xlabel('ROM Prediction (MPa)',fontsize=14)
        plt.ylabel('True Value from Simulation (MPa)',fontsize=14)
        plt.xlim([minV0,maxV0])
        plt.ylim([minV0,maxV0])
        figname = 'ROMsData-validation'+str(i+1)
        plt.savefig(figname,bbox_inches='tight')
        #plt.show()
        plt.close()
    plt.figure()
    plt.scatter(np.arange(1,nData+1,1),CorreCoef_data,marker='d',color='red')
    plt.xlabel('Monitoring Data Point',fontsize=14)
    plt.ylabel('Correlation Coefficient',fontsize=14)
    plt.xlim([0,nData+1])
    plt.ylim([0.95,1])
    plt.savefig("CorrelationCoeff_data",bbox_inches='tight')
    plt.show()
    
    # plot3
    plt.figure()
    plt.scatter(predict_obj_sum,y_train_sum,marker='*',color='blue')
    real_max1=max(y_train_sum)
    pred_max1=max(predict_obj_sum)
    maxV1=max(real_max1,pred_max1)+1000
    plt.plot([0,maxV1],[0,maxV1],ls="-",c="0.3")
    plt.xlabel('ROMs Prediction (Kg)',fontsize=14)
    plt.ylabel('True Value from Simulation (Kg)',fontsize=14)
    plt.xlim([0,maxV1])
    plt.ylim([0,maxV1])
    plt.savefig("ROMs-total",bbox_inches='tight')
    plt.show()

        
## Step 4: Construct the Mars ROMs for data and response of interest
# ROMs for data points
ROM_data = {}
for iData in range(0, nData):
    ROM_data[iData] = Earth()
    ROM_data[iData].fit(x_train_scaled[:nTrain], data_train[iData])
print 'Build the ROMs for data points: Done!'

# ROMs for obj
ROM_obj = {}
for iLeak in range(0,NLeakPotential):
    ROM_obj[iLeak] = Earth()
    ROM_obj[iLeak].fit(x_train_scaled[:nTrain],y_train[iLeak])
print 'Build the ROMs for objective of interests: Done!'    


## Step 5: Generate Monte Carlo(MC) samples
np.random.seed(787878)
mc_design = np.random.rand(nMCSamples, nParam)
mc_design = mc_design*2 - 1
print 'Generate Monte Carlo samples: Done!'


## Step 6: Evaluate the MC samples using the built ROMs for data points/objs
mc_data = np.zeros((nData, nMCSamples))
for iData in range(0, nData):
    mc_data[iData] = ROM_data[iData].predict(mc_design)

mc_obj = np.zeros((NLeakPotential, nMCSamples))    
for iLeak in range(0,NLeakPotential):
    mc_obj[iLeak] = ROM_obj[iLeak].predict(mc_design)
mc_obj_sum = sum(mc_obj,0)
for i in range(0,nMCSamples):
    if mc_obj_sum[i]<0:
        mc_obj_sum[i]=0
print 'Evaluate the MC samples: Done!'

## Step 7: Calculate posterior distribution and uncertainty reduction

prior_mean = sum(mc_obj_sum)/len(mc_obj_sum)
prior_p90mp10 = uncertaintyMetric(mc_obj_sum)

# Generate synthetic data
#synthetic_data = mc_data.T[:nDataRealization]
def fehm(p):
    print ''
p = matk(model=fehm)
p.add_par('perm4',min=1e-19,max=1e-14)
p.add_par('perm5',min=1e-19,max=1e-14)
p.add_par('perm6',min=1e-19,max=1e-14)
p.add_par('kmult',min=0.5, max=2.0)
s = p.lhs(siz=nDataRealization, seed=1000)
LHSamples=s.samples.values
LHSamples_scaled = np.zeros_like(LHSamples)
for i in range(0,nDataRealization):
    for j in range(0,nParam):
        LHSamples_scaled[i][j] = 2*(LHSamples[i][j] - x_min[j])/(x_max[j] - x_min[j]) - 1
synthetic_data_raw = np.zeros((nData, nDataRealization))
for iData in range(0, nData):
    synthetic_data_raw[iData] = ROM_data[iData].predict(LHSamples_scaled)
synthetic_data=synthetic_data_raw.T
print 'Generate synthetic monitoring data: Done!'

# Calculate posterior metrics
#post_p90mp10_mean, post_p90mp10_iData, post_mean_iData, nSamples = Uncertainty_calc(mc_data,synthetic_data,mc_obj,err_option,0)

post_p90mp10_mean, post_p90mp10_time, post_p90mp10_iData, post_mean, post_mean_iData, nSamples,mc_obj_sum_post = Uncertainty_calc(mc_data,synthetic_data,mc_obj_sum,err_option,eps,time_sensitivity)


print ''
print 'Data assimilation is done!'
print ''


## Step 8: Post-processing
if post_processing==1:
    print 'Post-processing'
    
    # plot posterior uncertainty change with time
    plt.figure()
    U = np.insert(post_p90mp10_time,0,prior_p90mp10)
    #T = np.insert(time_point,0,0)
    plt.plot(time_point,U,marker='+',color='red')
    plt.xlabel('Time (Days)',fontsize=14)
    plt.ylabel('P90-P10 Cumulative CO$_2$ Leak (Kg)',fontsize=14)
    plt.xlim([0,Total_time])
    plt.ylim([0,10000])
    plt.savefig("UR.png",bbox_inches='tight')
    plt.show()
    
    # plot U values of prior and posterior distributions
    plt.figure()
    plt.scatter(1,prior_p90mp10,marker='d',color='red')
    xl=np.zeros(nDataRealization)
    for i in range(0,nDataRealization):
        xl[i]= 2
    plt.scatter(xl,post_p90mp10_iData,marker='d',color='blue')
    plt.plot((0.7,2.3),(post_p90mp10_mean,post_p90mp10_mean),ls='--',color='blue')
    #plt.plot((1.1,1.1),(post_p90mp10_mean,prior_p90mp10),ls='-',color='red')
    plt.annotate('',xy=(1.1,post_p90mp10_mean),xycoords='data',xytext=(1.1,prior_p90mp10),textcoords='data',arrowprops={'arrowstyle':'<->','color':'red','lw':'1.5'})
    plt.figtext(0.45,0.7,"Uncertainty Reduction",fontsize=14, color='red')
    #plt.xlabel('',fontsize=14)
    plt.ylabel('P90-P10 Cumulative CO$_2$ Leak (Kg)',fontsize=14)
    plt.xticks(np.arange(4),('','Prior','Posterior',''))
    plt.ylim([0,10000])
    #plt.legend(loc='center right')
    plt.savefig("U of prior and posterior",bbox_inches='tight')
    plt.show()

    
    # plot CDF for prior and posterior distribution of obj
    plt.figure()
    num_bins =100
    hist, bin_edges = np.histogram(mc_obj_sum, bins=num_bins)
    cdf = np.cumsum(hist)
    cdf1=np.zeros(num_bins)
    for i in range(0,num_bins):
        cdf1[i]=float(cdf[i])/nMCSamples
    plt.plot(bin_edges[1:],cdf1,'r',label="Prior")
    for i in range(0,nDataRealization):
        hist_post, bin_edges_post = np.histogram(mc_obj_sum_post[i], bins=num_bins)
        cdf_post = np.cumsum(hist_post)
        cdf1_post=np.zeros(num_bins)
        for j in range(0,num_bins):
            cdf1_post[j]=float(cdf_post[j])/len(mc_obj_sum_post[i])
        if i==0:
            plt.plot(bin_edges_post[1:],cdf1_post,'b',label="Posterior")
        else:
            plt.plot(bin_edges_post[1:],cdf1_post,'b')
    plt.xlabel('Cumulative CO$_2$ Leak (Kg)',fontsize=14)
    plt.ylabel('CDF',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig("CDF.png",bbox_inches='tight')    
    plt.show()

    #plot histogram
    plt.figure()
    num_bins1=50
    plt.hist(mc_obj_sum,bins=num_bins1,color='blue',label='Prior')
    plt.hist(mc_obj_sum_post[0],bins=num_bins1,color='orange',label='Posterior_R1')
    plt.hist(mc_obj_sum_post[149],bins=num_bins1,color='red',label='Posterior_R150')
    plt.xlabel('Cumulative CO$_2$ Leak (Kg)',fontsize=14)   
    plt.ylabel('Frequency',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig("histogram.png",bbox_inches='tight')
    plt.show()
    
    # plot the data realization
    plt.figure()
    for i in range(0,nDataRealization):
        data_plot=np.insert(synthetic_data[i],0,data_train_raw[0][0])
        plt.plot(time_point,data_plot)
    plt.xlabel('Time (Days)',fontsize=14)
    plt.ylabel('Pressure at Monitoring Point (MPa)',fontsize=14)
    plt.xlim([0,Total_time])
    plt.savefig("data_realization.png",bbox_inches='tight')
    plt.show()
    
    # plot number of samples remained
    plt.figure()
    plt.scatter(np.arange(1,nDataRealization+1,1),nSamples,marker='o',color='blue')
    plt.xlabel('Data Realization',fontsize=14)
    plt.ylabel('Samples Remained',fontsize=14)
    plt.xlim([0,nDataRealization])
    #plt.ylim([0,nMCSamples])
    plt.savefig("samples_remained.png",bbox_inches='tight')
    plt.show()

    print ''
    print 'Workflow: done successfully!'
    print ''
    print ''
