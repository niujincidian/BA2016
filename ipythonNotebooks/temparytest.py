import os
import numpy as np
import os.path

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams.update({'font.size': 15})

from nideep.eval.learning_curve import LearningCurve
from nideep.eval.eval_utils import Phase

import nideep.eval.log_utils as lu
    # set up Python environment: numpy for numerical routines, and matplotlib for plotting
import h5py
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sensSpec
#plot roc_curve:
import roc_curve
from helper.cacheResults import listSave,listRead

print("Done importing")

soundTypes = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleSpeech', 'fire',
       'footsteps', 'knock', 'phone', 'piano']
print_soundTypes = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femS', 'fire',
       'ftStp', 'knock', 'phone', 'piano']

## this is the test net prototxt
## this is the test net prototxt
model_def = '/mnt/antares_raid/home/cindy/adhara/experiments/26/deploy.prototxt'
snapshot_prefix = '/mnt/raid/dnn/cindy/modelfiles/26/te'
it_AdaDel,sens_AdaDel,spec_AdaDel,bal_AdaDel = sensSpec.getSensSpecSCE(model_def,
                                                      soundTypes,
                                                      snapshot_prefix,
                                                      s_prefix= '/mnt/raid/dnn/cindy/modelfiles/26/te',
                                                      s_flag=True,
                                                                                              threshold=0.5,
                                                      test_size=23567,
                                                      batch_size=128,
                                                      snapshot=3000,
                                                      max_iter=99000)
saveFileName='/mnt/antares_raid/home/cindy/adhara/experiments/result/results26.npy'
tmpdata = np.array([sens_AdaDel,spec_AdaDel,bal_AdaDel])
np.save(saveFileName,tmpdata)

