import os
import numpy as np

def listSave(sens_list,spec_list,bal_list,expN,resultType='test',snapshot=5000,max_iter=100000,saveDIR='/mnt/antares_raid/home/cindy/records/tpr/'):
    '''
    :iter_list,sens_list,spec_list,bal_list:data to be saved
    :expN: number of experiment
    :resultType: <string> test on test dataset or test on train dataset
        'test'
        'train'
    :saveDIR: where to find the cache
    '''
    #check wether len(lists) corresponds to 'snapshot' and 'max_iter':
    listLen = len(sens_list)
    modelfileN = max_iter/snapshot
    if listLen!=modelfileN:
        raise Exception("Parameter 'snapshot' and 'max_iter' aren't consistent with length of sens_lists.")
    saveFileName = saveDIR+resultType+'_'+str(expN).zfill(2)+'.npy'
    data = np.array([sens_list,spec_list,bal_list])
    np.save(saveFileName,data)
    return True
def listRead(expN,resultType='test',snapshot=5000,max_iter=100000,saveDIR='/mnt/antares_raid/home/cindy/records/tpr/'):
    filename = saveDIR+resultType+'_'+str(expN).zfill(2)+'.npy'
    if os.path.isfile(filename):
        data = np.load(filename)
        sens_list = data[0]
        spec_list = data[1]
        bal_list = data[2]
        ## construct iter_list ##
        iter_list = [snapshot*(i+1) for i in range(max_iter/snapshot)]
        return iter_list,sens_list,spec_list,bal_list
    else:
        raise Exception('No such file w.r.t. your definition')