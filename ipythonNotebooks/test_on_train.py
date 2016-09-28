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
import h5py
import caffe
import sensSpec
from caffe.proto import caffe_pb2
from google.protobuf import text_format

soundTypes = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleSpeech', 'fire',
       'footsteps', 'knock', 'phone', 'piano']
print_soundTypes = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femS', 'fire',
       'footS', 'knock', 'phone', 'piano']
# this is the test net prototxt
model_def = '/mnt/antares_raid/home/cindy/adhara/experiments/17/deploy_train.prototxt'
# this is where your snapshots are stored, the same as the field defined in solver.prototxt
# during the sens&spec calculation procedure, it creates cache for these values 
# default directory for these cache files is the same as these .caffemodel files.
# but this can be modified in the code 'save_prefix'
snapshot_prefix = '/mnt/raid/dnn/cindy/modelfiles/17/te'
it_list_nn,sens_list_nn,spec_list_nn,bal_acclist_nn = sensSpec.getSensSpecNoSlicing(model_def,
                                                      soundTypes,
                                                      snapshot_prefix,
                                                      s_prefix='/mnt/raid/dnn/cindy/modelfiles/17/te_train',
                                                      s_flag=False,
                                                      test_size=417888,
                                                      batch_size=128,
                                                      snapshot=5000,
                                                      max_iter=100000)

