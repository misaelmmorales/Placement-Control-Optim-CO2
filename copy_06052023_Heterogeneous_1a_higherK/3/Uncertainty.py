import numpy as np
from uncertaintyMetric import *


def  Uncertainty_calc(mc_data,synthetic_data,objs,err_option,eps,MeasureType,time_sensitivity,nLoc):
    print ' Calculating uncertainty reduction'
    nObj = 1 # currently we only consider one obj
    array_shape = np.shape(synthetic_data)
    nDataRealization = array_shape[0]
    nDataPoint = array_shape[1]
    post_p90mp10_iData = np.zeros((nDataRealization,nObj))
    post_mean_iData = np.zeros((nDataRealization,nObj))
    nSamples_remained_end = np.zeros(nDataRealization)

    if time_sensitivity:
        datapoint_picker = np.arange(0,nDataPoint/nLoc,1)
        nStep = len(datapoint_picker)
    else:
        datapoint_picker = nDataPoint/nLoc
        nStep = 1
    
    post_p90mp10_iData_iStep = np.zeros((nDataRealization,nStep))
    post_mean_iData_iStep    = np.zeros((nDataRealization,nStep))
    post_p90mp10_time = np.zeros(nStep)
    mc_obj_sum_post = []
    
    for iDataRealization in range(0, nDataRealization):    
        print 'Calculating uncertainty reduction assuming realization #' + str(iDataRealization+1) + ' to be true'
        filtered_sample = Util_calc_hmerr(mc_data.T,synthetic_data[iDataRealization],err_option,eps,MeasureType,datapoint_picker,nLoc)   
    
        #  End point result
        nSamples_remained_end[iDataRealization] = np.shape(filtered_sample[nDataPoint/nLoc-1])[0]    
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


                             
def  Util_calc_hmerr(mc_result,obs_data,err_option,eps,MeasureType,datapoint_picker,nLoc):
    # Util_calc_hmerr
    err = mc_result - obs_data
    abs_err_unscaled = abs(err)
    eps_vector=np.zeros(len(abs_err_unscaled.T))
    abs_err=np.zeros_like(abs_err_unscaled)
    nTimeSeries=int(len(eps_vector)/nLoc)
    for i in range(0,nLoc):
        if (MeasureType==4) or (MeasureType==5) or (MeasureType==6) or (MeasureType==7):
            eps_vector[i*nTimeSeries:(i+1)*nTimeSeries]=eps[i]
        else:
            eps_vector[i*nTimeSeries:(i+1)*nTimeSeries]=eps
        	  
    for i in range(0,len(abs_err_unscaled)):
        abs_err[i]=abs_err_unscaled[i]/eps_vector
        
    filtered_sample = []
#    if err_option == 2:  # MeanAE
#      #  for i in range(0,len(datapoint_picker)):
#            err_mc = mean(abs_err[datapoint_picker[i]])/len(abs_err[datapoint_picker[i]])
#            if err_mc < eps:
#                filtered_sample = np.where(err_mc<eps)           
    if err_option == 3: # MaxAE
        abs_err_v=np.zeros_like(abs_err.T)
        for i in range(0, len(datapoint_picker)*nLoc):
            if i%nLoc == 0:
                abs_err_v[i] = abs_err.T[i/nLoc]
            elif i%nLoc == 1:
                abs_err_v[i] = abs_err.T[len(datapoint_picker)+(i-1)/nLoc]
            elif i%nLoc == 2:
                abs_err_v[i] = abs_err.T[len(datapoint_picker)*2+(i-2)/nLoc]
            else:
                print 'At most three monitoring wells can be handled, for more wells, please revise the code!'
                quit()
        for i in range(0,len(datapoint_picker)):
            err_mc = abs_err_v[:datapoint_picker[i]*nLoc+nLoc].max(axis=0)
            filtered_sample.append(np.where(err_mc<1)[0])
    else:
        print 'Wrong err_option'
        
    return filtered_sample 
