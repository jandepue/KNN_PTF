#==============================================================================
# Demonstration of kNN pedotransfer function
# Prediction for UNSODA dataset, using leave one out procedure + bootstrap
#
# Created By Jan De Pue (2020) at Ghent University
# References:
# Botula, Yves-Dady, et al. "Prediction of water retention of soils from the humid tropics by the nonparametric k‐nearest neighbor approach." Vadose Zone Journal 12.2 (2013): 1-17.
# De Pue et al.
#==============================================================================

import pylab
import numpy
import copy
import random
import sys
import os
import inspect

from KNN_Function import *
from Validation_Funk import *

from matplotlib.backends.backend_pdf import PdfPages
figlist=[]
figsize1=[10,10]
figsize2=[10,10./numpy.sqrt(2)]
figsize3=[10,50]
#dpi=None
dpi=300

font = {'family' : 'monospace',
        'size'   : 12}
params = {'mathtext.default': 'regular',
        }
pylab.rc('font', **font)
pylab.rcParams.update(params)
pylab.ioff()

cmap = pylab.get_cmap('Paired')

ScriptBasename = os.path.basename(inspect.getfile(inspect.currentframe()))[:-3]
figlist = []

#==============================================================================
# Specify Dataset
#==============================================================================

Label = 'Unsoda'
TrainFilename = 'SoilData/Unsoda_Export.txt'
InputFilename = 'SoilData/Unsoda_Export.txt'
IDCol = [0,]
InCol = [1,2,3,4] # predictor variables
OutCol = [9,10,11,12] # response variables
InLog = [] # colums to be logtransformed
OutLog = [] # colums to be logtransformed

print(Label)

#==============================================================================
# Open Data
#==============================================================================
print('Open Data')

fID=open(TrainFilename)
DataFields=numpy.array(fID.readline().replace('\r\n','\n').replace('\n','').split('\t'))
fID.close()

Header_ID=DataFields[IDCol]
Header_In=DataFields[InCol]
Header_Out=DataFields[OutCol]

InputData_ID = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(IDCol),skip_header=1,dtype='str')

# ## Option 1: Jacknife
# nS = InputData_ID.shape[0]
# nTrainSeparation = int(nS*0.7)

# if TrainFilename==InputFilename:
#     RandomSample = numpy.array(random.sample(range(nS),nTrainSeparation))
#     RandomFilt = numpy.zeros(nS,dtype='bool')
#     RandomFilt[RandomSample] = True

## Option 2: Leave One Out
LeaveOneOut = True
# LeaveOneOut = False


# Training data
if (TrainFilename==InputFilename) & (LeaveOneOut == False): # seperate the dataset in two
    TrainData_In_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(InCol),skip_header=1)[RandomFilt,:]
    TrainData_Out_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(OutCol),skip_header=1)[RandomFilt,:]
else:
    TrainData_In_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(InCol),skip_header=1)
    TrainData_Out_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(OutCol),skip_header=1)

# Input data
if (TrainFilename==InputFilename) & (LeaveOneOut == False): # seperate the dataset in two
    InputData_In_0 = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(InCol),skip_header=1)[~RandomFilt,:]
    InputData_Out_0=numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(OutCol),skip_header=1)[~RandomFilt,:]
else:
    InputData_In_0 = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(InCol),skip_header=1)
    InputData_Out_0=numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(OutCol),skip_header=1)


## Remove NaN
nanValue = -999
TrainData_In_0[TrainData_In_0 == nanValue] = numpy.nan
TrainData_Out_0[TrainData_Out_0 == nanValue] = numpy.nan
InputData_In_0[InputData_In_0 == nanValue] = numpy.nan
InputData_Out_0[InputData_Out_0 == nanValue] = numpy.nan
nanfilt = ~(numpy.any(numpy.isnan(TrainData_In_0),axis=1) | numpy.any(numpy.isnan(TrainData_Out_0),axis=1))
TrainData_In_0  = TrainData_In_0[nanfilt,:]
TrainData_Out_0 = TrainData_Out_0[nanfilt,:]

nanfilt = ~(numpy.any(numpy.isnan(InputData_In_0),axis=1) | numpy.any(numpy.isnan(InputData_Out_0),axis=1))
InputData_In_0  = InputData_In_0[nanfilt,:]
InputData_Out_0 = InputData_Out_0[nanfilt,:]


## Count
nTrain,nIn=TrainData_In_0.shape
nTrain,nOut=TrainData_Out_0.shape
nInput,nIn=InputData_In_0.shape
# nInput,nOut=InputData_Out_0.shape

## Log Transformations if needed
for iIn in InLog:
    TrainData_In_0[:,iIn] = numpy.log10(TrainData_In_0[:,iIn]+1)
    InputData_In_0[:,iIn] = numpy.log10(InputData_In_0[:,iIn]+1)

for iO in OutLog:
    TrainData_Out_0[:,iO] = numpy.log10(TrainData_Out_0[:,iO]+1)
    InputData_Out_0[:,iO] = numpy.log10(InputData_Out_0[:,iO]+1)

## Boxplot TrainingData
fig=pylab.figure()
fig.suptitle('Original')
ax=fig.add_subplot(211)
bp1=ax.boxplot(TrainData_In_0)
ax.set_xticklabels(Header_In)
ax.set_title('Training Data')
ax=fig.add_subplot(212)
bp1=ax.boxplot(InputData_In_0)
ax.set_xticklabels(Header_In)
ax.set_title('Input Data')
figlist.append(fig)

fig=pylab.figure()
fig.suptitle('Original')
ax=fig.add_subplot(211)
bp1=ax.boxplot(TrainData_Out_0)
ax.set_xticklabels(Header_Out)
ax.set_title('Training Data')
ax=fig.add_subplot(212)
bp1=ax.boxplot(InputData_Out_0)
ax.set_xticklabels(Header_Out)
ax.set_title('Input Data')
figlist.append(fig)


## Normalize
TrainData_In_mean = numpy.mean(TrainData_In_0,axis=0)
TrainData_In_std = numpy.std(TrainData_In_0,axis=0)

TrainData_In = (TrainData_In_0 - TrainData_In_mean)/(TrainData_In_std)
InputData_In = (InputData_In_0 - TrainData_In_mean)/(TrainData_In_std)
TrainData_Out = TrainData_Out_0
InputData_Out = InputData_Out_0



#==============================================================================
# kNN
#==============================================================================
print('kNN')

## BOOTSTRAPPING
nB = 10
BootSize=0.8
# nB = 1
# BootSize=1.0

KNNEst_Boot=numpy.empty((nInput,nOut,nB))
for iB in range(nB):
    # Bootstrap subsampling
    print("%s / %s"%(iB,nB-1))
    nSamp=int(nTrain*BootSize)# part of dataset used for resampling to measure statistics (0.935 or 0.8)
    RandUnique=random.sample(range(nTrain), nSamp) # Generate unique random numbers
    TrainData_In_Samp=TrainData_In[RandUnique,:]
    TrainData_Out_Samp=TrainData_Out[RandUnique,:]
    TrainData_In_0_Samp=TrainData_In_0[RandUnique,:]
    TrainData_Out_0_Samp=TrainData_Out_0[RandUnique,:]

    for iO in range(nOut):
        KNNEst_Boot[:,iO,iB] = KNN(TrainData_In_0_Samp,
                                   InputData_In_0,
                                   TrainData_Out_0_Samp[:,iO][:,None],
                                   LeaveOneOut = LeaveOneOut,
                                   Knumber = -1, # default: use equation by Botula et al.
                                   Power = -1, # default: use equation by Botula et al.
                                )[:,0]

KNN_Est = KNNEst_Boot.mean(axis=2)
KNN_Est_BootSTD = KNNEst_Boot.std(axis=2).mean(axis=0)
KNN_Est_BootSTD_All = KNNEst_Boot.std(axis=2)

#==============================================================================
# Validation
#==============================================================================

KNN_Error = KNN_Est - InputData_Out_0
KNN_RelError = KNN_Error/InputData_Out_0
KNN_ME = numpy.mean(KNN_Error,axis=0)
KNN_RMSE = numpy.sqrt(numpy.sum(KNN_Error**2,axis=0)/nInput)
KNN_RelRMSE = numpy.sqrt(numpy.sum(KNN_RelError**2,axis=0)/nInput)
KNN_R2 = PearsonR2(InputData_Out_0,KNN_Est,axis = 0)
KNN_NS = NashSutcliffeMEC(InputData_Out_0,KNN_Est,axis = 0)

#==============================================================================
# Plot
#==============================================================================

fig = pylab.figure(figsize = [8,8])
ax = fig.add_subplot(111)
for iO in range(nOut):
    color=cmap(iO/(nOut-0.99))
    # ax.plot(InputData_Out_0[:,iO], KNN_Est[:,iO],
    #         '.',color = color, label = Header_Out[iO])
    ax.errorbar(InputData_Out_0[:,iO], KNN_Est[:,iO],
                yerr = KNNEst_Boot.std(axis=2)[:,iO],
                fmt='.',color = color, label = Header_Out[iO])
xmin = min(ax.get_xlim()[0],ax.get_ylim()[0])
xmax = max(ax.get_xlim()[1],ax.get_ylim()[1])
xmin = xmin - abs(xmin)*0.1
xmax = xmax + abs(xmax)*0.1
ax.plot([xmin,xmax],[xmin,xmax],'-r')
ax.set_xlim([xmin,xmax])
ax.set_ylim([xmin,xmax])
ax.set_aspect('equal')
ax.set_xlabel('True WC (m3/m3)')
ax.set_ylabel('Predicted WC(m3/m3)')
ax.set_title('kNN')

ax.legend(loc = 4)
figlist.append(fig)

#==============================================================================
# Write
#==============================================================================

print('Writing data'.center(50,'='))

import sys
basename=ScriptBasename
postfix = ''

# save plots to pdf
pdfname = '%s%s.pdf'%(basename,postfix)
pp = PdfPages(pdfname)
for fig in figlist:
    pp.savefig(fig)
pp.close()

pylab.show()

