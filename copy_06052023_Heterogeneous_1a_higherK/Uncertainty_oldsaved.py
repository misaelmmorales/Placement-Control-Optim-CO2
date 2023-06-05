import numpy as np
from uncertaintyMetric import *


def  Uncertainty_calc(mc_data,synthetic_data,objs,err_option,eps,time_sensitivity):
    print ' Calculating uncertainty reduction'
    nObj = 1 # currently we only consider one obj
    array_shape = np.shape(synthetic_data)
    nDataRealization = array_shape[0]
    nDataPoint = array_shape[1]
    post_p90mp10_iData = np.zeros((nDataRealization,nObj))
    post_mean_iData = np.zeros((nDataRealization,nObj))
    nSamples_remained_end = np.zeros(nDataRealization)

    if time_sensitivity:
        datapoint_picker = np.arange(0,nDataPoint,1)
        nStep = len(datapoint_picker)
    else:
        datapoint_picker = nDataPoint
        nStep = 1
    
    post_p90mp10_iData_iStep = np.zeros((nDataRealization,nStep))
    post_mean_iData_iStep    = np.zeros((nDataRealization,nStep))
    post_p90mp10_time = np.zeros(nStep)
    mc_obj_sum_post = []
    
    for iDataRealization in range(0, nDataRealization):    
        print 'Calculating uncertainty reduction assuming realization #' + str(iDataRealization+1) + ' to be true'
        filtered_sample = Util_calc_hmerr(mc_data.T,synthetic_data[iDataRealization],err_option,eps,datapoint_picker)   
    
        #  End point result
        nSamples_remained_end[iDataRealization] = np.shape(filtered_sample[nDataPoint-1])[0]    
        if nSamples_remained_end[iDataRealization]<50:
            print 'Warning, number of samples smaller than 30, ignore realization #' + str(iDataRealization+1)
            post_p90mp10_iData[iDataRealization] = -1
            post_mean_iData[iDataRealization] = -1
        else:
            post_p90mp10_iData[iDataRealization] = uncertaintyMetric(objs[filtered_sample[nStep-1]])
            post_mean_iData[iDataRealization] = sum(objs[filtered_sample[nStep-1]])/nSamples_remained_end[iDataRealization]
            mc_obj_sum_post.append(objs[filtered_sample[nStep-1]])
        # Time dependent result
        for iStep in range(0,nStep):
            nSamples_remained_iStep = len(filtered_sample[iStep])
            if nSamples_remained_iStep<50:
                post_p90mp10_iData_iStep[iDataRealization][iStep] = -1
                post_mean_iData_iStep[iDataRealization][iStep] = -1
            else:
                post_p90mp10_iData_iStep[iDataRealization][iStep] = uncertaintyMetric(objs[filtered_sample[iStep]])
                post_mean_iData_iStep[iDataRealization][iStep] = sum(objs[filtered_sample[iStep]])/ len(filtered_sample[iStep])
                             
    for iStep in range(0, nStep):
        temp1 = post_p90mp10_iData_iStep.T[iStep]
        temp1 = [element for element in temp1 if element>0]
        post_p90mp10_time[iStep] = sum(temp1)/len(temp1)
    post_p90mp10_mean = post_p90mp10_time[nStep-1]

    temp2 = post_mean_iData
    temp2 = [element for element in temp2 if element>0]
    post_mean = sum(temp2)/len(temp2)
                             
    return post_p90mp10_mean,post_p90mp10_time,post_p90mp10_iData,post_mean,post_mean_iData,nSamples_remained_end,mc_obj_sum_post


                             
def  Util_calc_hmerr(mc_result,obs_data,err_option,eps,datapoint_picker):
    # Util_calc_hmerr
    err = mc_result - obs_data
    abs_err = abs(err)
    filtered_sample = []
#    if err_option == 2:  # MeanAE    
#      #  for i in range(0,len(datapoint_picker)):
#            err_mc = mean(abs_err[datapoint_picker[i]])/len(abs_err[datapoint_picker[i]])
#            if err_mc < eps:
#                filtered_sample = np.where(err_mc<eps)           
    if err_option == 3: # MaxAE
        for i in range(0,len(datapoint_picker)):
            err_mc = abs_err.T[:datapoint_picker[i]+1].max(axis=0)
            filtered_sample.append(np.where(err_mc<eps)[0])        
    else:
        print 'Wrong err_option'
        
    return filtered_sample 



