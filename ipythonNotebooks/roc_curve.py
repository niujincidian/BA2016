import h5py
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

def multi_class_1hot(fpath, key_est, key_label='label', classes=None):
    
    f = h5py.File(fpath, 'r')
    
    ip = f[key_est][:]
    l = f[key_label][:]
    if classes is None:
        classes = range(ip.shape[-1])
    
    for cl, clname in enumerate(classes):
        fpr, tpr, _ = metrics.roc_curve(np.squeeze(l==cl).ravel(),
                                        ip[:, cl].ravel())
        #plt.subplot(1, len(class_names), cl+1)
        plt.plot(fpr, tpr, label='%s ROC (auc=%2f)' % (clname, metrics.auc(fpr, tpr),))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title("ROC curves (multi-class 1-hot)")
        plt.legend(loc='lower right')
    
def multi_class(fpath, key_est, key_label='label', classes=None):
    
    f = h5py.File(fpath, 'r')
    
    ip = f[key_est][:]
    l = np.squeeze(f[key_label][:])
    if classes is None:
        classes = range(ip.shape[-1])
    
    for cl, clname in enumerate(classes):
        fpr, tpr, thresholds = metrics.roc_curve(l[:, cl].ravel(),
                                        ip[:, cl].ravel())
        #print thresholds 
        #plt.subplot(1, len(class_names), cl+1)
        plt.plot(fpr, tpr, label='%s ROC (auc=%2f)' % (clname, metrics.auc(fpr, tpr),))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title("ROC curves (multi-class)")
        plt.legend(loc='lower right')
    return fpr,tpr,thresholds
        
if __name__ == '__main__':
    fpath_pred =  '/home/kashefy/models/sound/twoears/x4/test_0.h5'
    class_names = ['alarm',
                   'baby',
                   'crash',
                   'dog',
                   'engine',
                   'femaleSpeech',
                   'fire',
                   'footsteps',
                   'knock',
                   'phone',
                   'piano'
                   ]
    
    key_est = 'ip3'
    plt.figure()
    multi_class(fpath_pred, key_est, key_label='label', classes=class_names)

    plt.show()