###############################################
## Sensitivity and Specificity calculation
## Specialized for 3 types of dnn architectures
###############################################
import caffe
import os
import numpy as np
import h5py
import nideep.eval.inference
import helper

# global variable for calculation cache
cacheDIR = '/mnt/antares_raid/home/cindy/records/tpr/'

def __getSensSpecListsSCE(netFile,modelfilename,cachepath,threshold=0.5,batch_size=128,N=11,test_size=23567):
    '''
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :modelfilename: [string] <.caffemodel> snapshots obtained from training process
                    they are the trained weights for each layer with weights
    :N: [int] number of classification classes
    :test_size: specifies size of test dataset
    :batch_size: test batch_size
    '''
    # if there are multiple gpu devices, choose the first one 
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    ### Check how many forward operations to take in order to go through all the test data set
    test_iter = np.ceil(test_size*1.0/batch_size).astype(int)  
    # print str(test_iter)+' iterations required'
    
    ########################### check and do the forward(),save the outputs to cache ###########################
    if not(os.path.isfile(cachepath)):
        # load train_val prototxt
        net = caffe.Net(netFile,      # defines the structure of the model
                    modelfilename,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # print 'Done loading weights from '+modelfilename
        nideep.eval.inference.infer_to_h5_fixed_dims(net, ['label','ip3','predict'], test_iter, cachepath)
    ######################################################################################################
    
    cachedata = h5py.File(cachepath,'r')
    
    total = batch_size*test_iter

    # record lists initialization
    positive_true = np.zeros(N)
    negative_true = np.zeros(N)
    thresholding = np.ones((total,N))*threshold
    # print 'threshold:',threshold
    # get values from sigmoid function
    # apply thresholding:
    predict_data = cachedata['predict']
    prediction = (predict_data>thresholding).astype(int)
    label = cachedata['label'][:,0,:,0].astype(int)
    positive_amount = np.sum(label,axis=0)
    #count positive-true:
    compare_matrix = np.multiply(prediction,label)
    positive_true = np.sum(compare_matrix,axis=0)
    #count negative-true:
    neg_prediction = np.logical_not(prediction).astype(int)
    neg_label = np.logical_not(label).astype(int)
    neg_compare_matrix = np.multiply(neg_prediction,neg_label)
    negative_true = np.sum(neg_compare_matrix,axis=0)
    
    sensitivity = positive_true*1.0/positive_amount
    negative_amount = total*np.ones(N)-positive_amount
    specificity = negative_true*1.0/negative_amount
    balanced_accuracy = (sensitivity+specificity)/2
    
    return sensitivity,specificity,balanced_accuracy

def getSensSpecSCE(netFile,soundTypes,snapshot_prefix,s_prefix,s_flag=True,threshold=0.5,batch_size=128, test_size = 23567,snapshot=5000,max_iter=100000):
    '''
    [Input]
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :soundTypes: list of class names
    :s_flag: True for storing the cache in the same directory of the .caffemodel files
    :batch_size: test data batch_size
        if the batch_size too big, it gives an error
        128 is a good choice
    :snapshot_prefix,
    snapshot,
    max_iter: the same as those in solver.prototxt
    :s_prefix: save directory
    [Output]
    2-dim arrays:
    :sensitivity_list:snapshotsNum x K array
    :specificity_list:snapshotsNum x K array
    :balanced_accuracy_list:snapshotsNum x K array
    snapshotsNum:number of snapshots used for plotting
    K: number of classes
    [.npy file format]
    save the calculated acc_values to .npy file:
        1. sensitivity
        2. specificity
        3. balanced_accuracy
    '''
    ### get test data file and check the size of test dataset
    ##TODO##
    # parse the .prototxt file
    print 'enter function'
    # get number of classes
    N = len(soundTypes)
    ### read in modelfile lists:
    # number of snapshots
    snapshotsNum = max_iter/snapshot
    print snapshotsNum,'model files'
    sensitivity_list = []
    specificity_list = []
    balanced_accuracy_list = []
    iternum_list = []
    for k in range(snapshotsNum):
        iternum = snapshot*(k+1)
        iternum_list.append(iternum)
        prefix = snapshot_prefix + '_iter_'+str(iternum)
        if s_flag:
            save_prefix = prefix
        else:
            save_prefix = s_prefix + '_iter_'+str(iternum)
        cachepath = save_prefix+'.hdf5'
        model_temp = prefix+'.caffemodel'
        sensitivity,specificity,balanced_accuracy=__getSensSpecListsSCE(netFile,
                                                                        model_temp,
                                                                        cachepath,
                                                                        threshold,
                                                                        batch_size,
                                                                        N,
                                                                        test_size)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        balanced_accuracy_list.append(balanced_accuracy)
#         print 'sensitivity',sensitivity
#         print 'specificity',specificity
#         print 'balanced_accuracy',balanced_accuracy
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    balanced_accuracy_list = np.array(balanced_accuracy_list)
    
    return iternum_list,sensitivity_list,specificity_list,balanced_accuracy_list

def __getSensSpecLists(netFile,modelfilename,cachepath,batch_size=128,N=11,test_size=136236):
    '''
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :modelfilename: [string] <.caffemodel> snapshots obtained from training process
                    they are the trained weights for each layer with weights
    :N: [int] number of classification classes
    :test_size: specifies size of test dataset
    :batch_size: test batch_size
    '''
    # if there are multiple gpu devices, choose the first one 
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    ### Check how many forward operations to take in order to go through all the test data set
    test_iter = np.ceil(test_size*1.0/batch_size).astype(int)  
    
    ### There are N classes
    predict_name_list = []
    label_name_list = []
    # this list stores the number of test data_points in each class
    for i in range(N):
        predict_name_list.append('predict'+str(i+1).zfill(2))
        label_name_list.append('label'+str(i+1).zfill(2))
    outputblobs = predict_name_list+label_name_list
    ########################### check and do the forward(),save the outputs to cache ###########################
    if not(os.path.isfile(cachepath)):
        # load train_val prototxt
        net = caffe.Net(netFile,      # defines the structure of the model
                    modelfilename,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # print 'Done loading weights from '+modelfilename
        nideep.eval.inference.infer_to_h5_fixed_dims(net, outputblobs, test_iter, cachepath)
    ######################################################################################################
    
    cachedata = h5py.File(cachepath,'r')
    
    test_amount = [0 for i in range(N)]
    true_positive_list = [0 for i in range(N)]
    true_negative_list = [0 for i in range(N)]
    for k in range(N):
        temp_output = cachedata[predict_name_list[k]]
        true_label = cachedata[label_name_list[k]][:,0,0,0].astype(bool)
        test_amount[k] += sum(true_label)
        predicted_label = (temp_output[:,0]<temp_output[:,1]).astype(bool)
        # for sensitivity
        true_positive = [x&y for (x,y) in zip(predicted_label, true_label)]
        true_positive_list[k] += sum(true_positive)
        # for specificity
        neg_label = [(not x) for x in true_label]
        neg_predicted_label = [(not x) for x in predicted_label]
        true_negative = [x&y for (x,y) in zip(neg_label, neg_predicted_label)]
        true_negative_list[k] += sum(true_negative)
    true_positive_list = np.array(true_positive_list)
    test_amount = np.array(test_amount)
    true_negative_list = np.array(true_negative_list)
    neg_test_amount = np.ones(N)*test_iter*128-test_amount

    sensitivity_list = np.divide(true_positive_list*1.0,test_amount)
    specificity_list = np.divide(true_negative_list*1.0,neg_test_amount)
    balanced_accuracy = (sensitivity_list+specificity_list)/2
    
    return sensitivity_list,specificity_list,balanced_accuracy

def getSensSpec(netFile,soundTypes,snapshot_prefix,s_prefix,s_flag=True,batch_size=128, test_size = 136236,snapshot=5000,max_iter=100000):
    '''
    [Input]
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :soundTypes: list of class names
    :s_flag: True for storing the cache in the same directory of the .caffemodel files
    :batch_size: test data batch_size
        if the batch_size too big, it gives an error
        128 is a good choice
    :snapshot_prefix,
    snapshot,
    max_iter: the same as those in solver.prototxt
    :s_prefix: save directory
    [Output]
    2-dim arrays:
    :sensitivity_list:snapshotsNum x K array
    :specificity_list:snapshotsNum x K array
    :balanced_accuracy_list:snapshotsNum x K array
    snapshotsNum:number of snapshots used for plotting
    K: number of classes
    [.npy file format]
    save the calculated acc_values to .npy file:
        1. sensitivity
        2. specificity
        3. balanced_accuracy
    '''
    ### get test data file and check the size of test dataset
    ##TODO##
    # parse the .prototxt file

    # get number of classes
    N = len(soundTypes)
    ### read in modelfile lists:
    # number of snapshots
    snapshotsNum = max_iter/snapshot
    print snapshotsNum,'model files'
    sensitivity_list = []
    specificity_list = []
    balanced_accuracy_list = []
    iternum_list = []
    for k in range(snapshotsNum):
        iternum = snapshot*(k+1)
        iternum_list.append(iternum)
        prefix = snapshot_prefix + '_iter_'+str(iternum)
        ##################################################
        # do the cache
        if s_flag:
            save_prefix = prefix
        else:
            save_prefix = s_prefix + '_iter_'+str(iternum)
        cachepath = save_prefix+'.hdf5'
        model_temp = prefix+'.caffemodel'
        
        sensitivity,specificity,balanced_accuracy = __getSensSpecLists(netFile,model_temp,cachepath,batch_size,N,test_size)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        balanced_accuracy_list.append(balanced_accuracy)
        #print 'sensitivity',sensitivity
        #print 'specificity',specificity
        #print 'balanced_accuracy',balanced_accuracy
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    balanced_accuracy_list = np.array(balanced_accuracy_list)
    
    return iternum_list,sensitivity_list,specificity_list,balanced_accuracy_list
'''
[notice]!
when trying to get the sens&spec values from net without slicing
use a different .prototxt file(deploy.prototxt)
in this file, there should be a 'Softmax' layer
just add this to your <deploy.prototxt> file
--------------------------------------------
layer{
  name: "prediction"
  type: "Softmax"
  bottom: "ip3"
  top: "prediction"
  include{
      phase: TEST
  }
}
---------------------------------------------
'''
def __getSensSpecListsNoSlicing(netFile,modelfilename,cachepath,batch_size=128,N=11,test_size=136236):
    '''
    --------no slicing version--------
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :modelfilename: [string] <.caffemodel> snapshots obtained from training process
                    they are the trained weights for each layer with weights
    :N: [int] number of classification classes
    :test_size: specifies size of test dataset
    :batch_size: test batch_size
    '''
    # if there are multiple gpu devices, choose the first one 
    caffe.set_device(0)
    caffe.set_mode_gpu()

    test_iter = np.ceil(test_size*1.0/batch_size).astype(int)  
    
    ########################### check and do the forward(),save the outputs to cache ###########################
    if not(os.path.isfile(cachepath)):
        # load train_val prototxt
        net = caffe.Net(netFile,      # defines the structure of the model
                    modelfilename,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # print 'Done loading weights from '+modelfilename
        nideep.eval.inference.infer_to_h5_fixed_dims(net, ['label_scalar','ip3'], test_iter, cachepath)
    ######################################################################################################
    
    cachedata = h5py.File(cachepath,'r')
    
    total = batch_size*test_iter
    
    K = N # 11 classes
    # record lists initialization
    positive_amount = np.zeros(K)
    positive_true = np.zeros(K)
    negative_true = np.zeros(K)
    prediction = np.argmax(cachedata['ip3'],axis=1)
    label_scalar = cachedata['label_scalar'][:,0,0,0].astype(int)
    for k in range(8):
        temp_vec = np.ones(total)*(k)
        label_k = (temp_vec==label_scalar)
        positive_amount[k]+=sum(label_k)
        # count true positive
        predict_k = (temp_vec==prediction)
        positive_true_k = np.multiply(predict_k,label_k)
        positive_true[k]+=sum(positive_true_k) 
        # count true negative
        neg_label_k = [(not x) for x in label_k]
        neg_predict_k = [(not x) for x in predict_k]
        negative_true_k = np.multiply(neg_label_k,neg_predict_k)
        negative_true[k]+=sum(negative_true_k)
    for k in range(8,11):
        temp_vec = np.ones(total)*(k+1)
        label_k = (temp_vec==label_scalar)
        positive_amount[k]+=sum(label_k)
        # count true positive
        predict_k = (temp_vec==prediction)
        positive_true_k = np.multiply(predict_k,label_k)
        positive_true[k]+=sum(positive_true_k) 
        # count true negative
        neg_label_k = [(not x) for x in label_k]
        neg_predict_k = [(not x) for x in predict_k]
        negative_true_k = np.multiply(neg_label_k,neg_predict_k)
        negative_true[k]+=sum(negative_true_k)
        
    sensitivity = positive_true*1.0/positive_amount
    negative_amount = total*np.ones(K)-positive_amount
    specificity = negative_true*1.0/negative_amount
    balanced_accuracy = (sensitivity+specificity)/2
    
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)
    balanced_accuracy = np.array(balanced_accuracy)
    return sensitivity,specificity,balanced_accuracy

def getSensSpecNoSlicing(netFile,soundTypes,snapshot_prefix,s_prefix,s_flag=True,test_size = 136236,batch_size=128,snapshot=5000,max_iter=100000):
    '''
    [Input]
    :netFile: [string] <deploy.prototxt> file used for forward operation
            in this file user must specify the source of training data set in data layer 
    :soundTypes: list of class names
    :s_flag: True for storing the cache in the same directory of the .caffemodel files
    :batch_size: test data batch_size
        if the batch_size too big, it gives an error
        128 is a good choice
    :snapshot_prefix,
    snapshot,
    max_iter: the same as those in solver.prototxt
    [Output]
    2-dim arrays:
    :sensitivity_list:snapshotsNum x K array
    :specificity_list:snapshotsNum x K array
    :balanced_accuracy_list:snapshotsNum x K array
    snapshotsNum:number of snapshots used for plotting
    K: number of classes
    [.npy file format]
    save the calculated acc_values to .npy file:
        1. sensitivity
        2. specificity
        3. balanced_accuracy
    '''
    ### get test data file and check the size of test dataset
    ##TODO##
    # parse the .prototxt file

    # get number of classes
    N = len(soundTypes)
    ### read in modelfile lists:
    # number of snapshots
    snapshotsNum = max_iter/snapshot
    print snapshotsNum,'model files'
    sensitivity_list = []
    specificity_list = []
    balanced_accuracy_list = []
    iternum_list = []
    for k in range(snapshotsNum):
        iternum = snapshot*(k+1)
        iternum_list.append(iternum)
        prefix = snapshot_prefix + '_iter_'+str(iternum)
        if s_flag:
            save_prefix = prefix
        else:
            save_prefix = s_prefix + '_iter_'+str(iternum)
        cachepath = save_prefix+'.hdf5'
        ################check whether the acc_values already exist######################
        model_temp = prefix+'.caffemodel'
        sensitivity,specificity,balanced_accuracy = __getSensSpecListsNoSlicing(netFile,
                                                                                    model_temp,
                                                                                cachepath,
                                                                                    batch_size,
                                                                                    N,
                                                                                    test_size)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        balanced_accuracy_list.append(balanced_accuracy)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    balanced_accuracy_list = np.array(balanced_accuracy_list)
    
    return iternum_list,sensitivity_list,specificity_list,balanced_accuracy_list

###########################################################
## some procedures useful for displaying the results
###########################################################
def printResult(print_soundTypes,sens,spec,bal):
    print 'class \t sensitivity \t specificity \t bal_acc'
    for s in range(len(print_soundTypes)):
        print '%s \t %f \t %f \t %f'%(print_soundTypes[s],sens[-1,s],spec[-1,s],bal[-1,s])