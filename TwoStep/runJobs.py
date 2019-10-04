#!/usr/bin/env python
from os import system, path, getcwd
from otherHelpers import submitJob
import numpy as np

#Set for testing command
dryRun = False
#dryRun = True

#Set for local submission
runLocal = True
#runLocal = False

myDir = getcwd()
baseDir = '/vols/cms/jwd18/Stage1categorisation/Pass1'
#years = ['2016','2017']

years = ['2016']
intLumi = 35.9

#years = ['2017']
#intLumi = 41.5

#To get a general idea of Hps, We are going to test 30 scenarios for three hyperparameters. 
#Each will have a "low", "medium", and "high" value, leading to
# 29 combinations, plus a test scenario with default values.
# eta = 0.05, 0.1, 0.25
# lambda = 0, 0.1, 0.2 
# max_depth = 3,10,20

#paramSets = [None]
#paramSets = ['eta:0.05,lambda:0,max_depth:3','eta:0.05,lambda:0,max_depth:10','eta:0.05,lambda:0,max_depth:20','eta:0.05,lambda:0.1,max_depth:3','eta:0.05,lambda:0.1,max_depth:10','eta:0.05,lambda:0.1,max_depth:20','eta:0.05,lambda:0.2,max_depth:3','eta:0.05,lambda:0.2,max_depth:10','eta:0.05,lambda:0.2,max_depth:20',
#'eta:0.1,lambda:0,max_depth:3','eta:0.1,lambda:0,max_depth:10','eta:0.1,lambda:0,max_depth:20','eta:0.1,lambda:0.1,max_depth:3','eta:0.1,lambda:0.1,max_depth:10','eta:0.1,lambda:0.1,max_depth:20','eta:0.1,lambda:0.2,max_depth:3','eta:0.1,lambda:0.2,max_depth:10','eta:0.1,lambda:0.2,max_depth:20',
#'eta:0.25,lambda:0,max_depth:3','eta:0.25,lambda:0,max_depth:10','eta:0.25,lambda:0,max_depth:20','eta:0.25,lambda:0.1,max_depth:3','eta:0.25,lambda:0.1,max_depth:10','eta:0.25,lambda:0.1,max_depth:20','eta:0.25,lambda:0.2,max_depth:3','eta:0.25,lambda:0.2,max_depth:10','eta:0.25,lambda:0.2,max_depth:20']


#NOTE: random HP search for dipho
'''
script    = 'diphotonCategorisation.py'
#param sets chosen randomly and submitted
#paramSets = []
#nIters = 2000
#nParams = 7

#for i in range(nIters):
#  paramSets.append('eta:%.2f' % np.random.uniform(0.05, 0.8))
#  paramSets.append('gamma:%.2f' % np.random.uniform(0,5))
#  paramSets.append('max_depth:%g' %np.random.randint(3,15))  
#  paramSets.append('subsample:%.2f' %np.random.uniform(0.6,1))  
#  paramSets.append('min_child_weight:%g' %np.random.randint(0,10))  

#paramSets = [
#  ",".join(paramSets[i:i+nParams])
#  for i in xrange(0, len(paramSets), nParams)
#]

paramSets = ['min_child_weight:5,subsample:0.847204,eta:0.75,max_depth:7,gamma:2.05'] #bestHPs
models = None
classModel = None
#dataFrame = 'trainTotal.pkl'
dataFrame = None
sigFrame  = None
#NOTE:end of dipho random opt


#NOTE: Bayesian optimiser for diphoton BDT. HP ranges specified in the script
#script = 'BayesDiphotonCategorisation.py'
#paramSets = [None]
#models    = None
#classModel = None
#dataFrame = 'trainTotal.pkl'
##dataFrame = None
#sigFrame  = None
'''

#NOTE: standard nJet BDT
'''
script    = 'nJetCategorisation.py'
paramSets =['min_child_weight:2,n_estimators:207,subsample:0.8920,eta:0.72,colsample_bytree:0.9,max_depth:14,gamma:0.01,lambda:0.9671'] #Best w_mc HPs
#paramSets =['min_child_weight:7,n_estimators:133,subsample:0.8151,eta:0.65,colsample_bytree:0.85,max_depth:8,gamma:0.08,lambda:0.1662'] #Best w_eq HPs
#paramSets =['min_child_weight:4,n_estimators:195,subsample:0.9518,eta:0.65,colsample_bytree:0.72,max_depth:14,gamma:0.01,lambda:0.4570'] #Best w_sqrt(eq) HPs
#paramSets =['min_child_weight:2,n_estimators:188,subsample:0.8661,eta:0.74,colsample_bytree:0.90,max_depth:9,gamma:0.04,lambda:0.3694'] #Best w_cbrt(eq) HPs
#paramSets =['min_child_weight:9,n_estimators:21,subsample:0.8943,eta:0.53,colsample_bytree:0.97,max_depth:7,gamma:2.00,lambda:0.7469'] #Best w_None HPs
#paramSets = [None]
#NB: n_estimators chosen through cross validation
#paramSets = [None,'eta:0.05,lambda:0,max_depth:3','eta:0.05,lambda:0,max_depth:10','eta:0.05,lambda:0,max_depth:20','eta:0.05,lambda:0.1,max_depth:3','eta:0.05,lambda:0.1,max_depth:10','eta:0.05,lambda:0.1,max_depth:20','eta:0.05,lambda:0.2,max_depth:3','eta:0.05,lambda:0.2,max_depth:10','eta:0.05,lambda:0.2,max_depth:20',
#'eta:0.1,lambda:0,max_depth:3','eta:0.1,lambda:0,max_depth:10','eta:0.1,lambda:0,max_depth:20','eta:0.1,lambda:0.1,max_depth:3','eta:0.1,lambda:0.1,max_depth:10','eta:0.1,lambda:0.1,max_depth:20','eta:0.1,lambda:0.2,max_depth:3','eta:0.1,lambda:0.2,max_depth:10','eta:0.1,lambda:0.2,max_depth:20',
#'eta:0.25,lambda:0,max_depth:3','eta:0.25,lambda:0,max_depth:10','eta:0.25,lambda:0,max_depth:20','eta:0.25,lambda:0.1,max_depth:3','eta:0.25,lambda:0.1,max_depth:10','eta:0.25,lambda:0.1,max_depth:20','eta:0.25,lambda:0.2,max_depth:3','eta:0.25,lambda:0.2,max_depth:10','eta:0.25,lambda:0.2,max_depth:20']
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#NOTE: bayes opt nJet BDT
'''
script    = 'nJetCategorisationWithBayesOpt.py'
paramSets = [None]
#NB: n_estimators chosen through cross validation
#paramSets = [None,'eta:0.05,lambda:0,max_depth:3','eta:0.05,lambda:0,max_depth:10','eta:0.05,lambda:0,max_depth:20','eta:0.05,lambda:0.1,max_depth:3','eta:0.05,lambda:0.1,max_depth:10','eta:0.05,lambda:0.1,max_depth:20','eta:0.05,lambda:0.2,max_depth:3','eta:0.05,lambda:0.2,max_depth:10','eta:0.05,lambda:0.2,max_depth:20',
#'eta:0.1,lambda:0,max_depth:3','eta:0.1,lambda:0,max_depth:10','eta:0.1,lambda:0,max_depth:20','eta:0.1,lambda:0.1,max_depth:3','eta:0.1,lambda:0.1,max_depth:10','eta:0.1,lambda:0.1,max_depth:20','eta:0.1,lambda:0.2,max_depth:3','eta:0.1,lambda:0.2,max_depth:10','eta:0.1,lambda:0.2,max_depth:20',
#'eta:0.25,lambda:0,max_depth:3','eta:0.25,lambda:0,max_depth:10','eta:0.25,lambda:0,max_depth:20','eta:0.25,lambda:0.1,max_depth:3','eta:0.25,lambda:0.1,max_depth:10','eta:0.25,lambda:0.1,max_depth:20','eta:0.25,lambda:0.2,max_depth:3','eta:0.25,lambda:0.2,max_depth:10','eta:0.25,lambda:0.2,max_depth:20']
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
#NOTE: here onwards is random HP search for nJet
script    = 'nJetCategorisation.py'
#param sets chosen randomly and submitted
paramSets = []
nIters = 3000
nParams = 7

for i in range(nIters):
  paramSets.append('eta:%.2f' % np.random.uniform(0.05, 0.8))
  paramSets.append('gamma:%.2f' % np.random.uniform(0,2))
  paramSets.append('colsample_bytree:%.2f' % np.random.uniform(0.5,1))
  paramSets.append('max_depth:%g' % np.random.randint(3,15))  
  paramSets.append('subsample:%f' % np.random.uniform(0.8,1))  
  paramSets.append('min_child_weight:%g' % np.random.randint(2,15))  
  paramSets.append('lambda:%f' % np.random.uniform(0.1,2))

paramSets = [
  ",".join(paramSets[i:i+nParams])
  for i in xrange(0, len(paramSets), nParams)
]

models = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
#NOTE:end of n jet random opt
'''

#NOTE: normal nClass BDT
'''
script    = 'nClassBDT.py'
#paramSets =['min_child_weight:13,n_estimators:18,sub_sample:0.9030,eta:0.76,colsample_bytree:0.95,max_depth:12,gamma:1.85,lambda:1.8283'] #best for MC weights 
#paramSets =['min_child_weight:2,n_estimators:47,sub_sample:0.9614,eta:0.74,colsample_bytree:0.98,max_depth:6,gamma:0.03,lambda:1.1377'] #best for EQ weights 
#paramSets =['min_child_weight:2,n_estimators:196,sub_sample:0.8505,eta:0.75,colsample_bytree:0.90,max_depth:9,gamma:0.08,lambda:0.9049'] #best for Sqrt(EQ) weights 
#paramSets =['min_child_weight:2,n_estimators:245,sub_sample:0.8985,eta:0.63,colsample_bytree:0.99,max_depth:14,gamma:0.06,lambda:0.9890'] #best for Cbrt(EQ) weights 
#paramSets =['min_child_weight:11,n_estimators:16,sub_sample:0.9683,eta:0.69,colsample_bytree:0.97,max_depth:8,gamma:0.91,lambda:0.2412'] #best for No weights 
#paramSets =['min_child_weight:12,n_estimators:17,sub_sample:0.9030,eta:0.7,colsample_bytree:0.95,max_depth:13,gamma:1.95,lambda:1.80'] # test: slightly peturb best model for MC weights 
#paramSets = [None]
#paramSets = [None,'eta:0.05,lambda:0,max_depth:3','eta:0.05,lambda:0,max_depth:10','eta:0.05,lambda:0,max_depth:20','eta:0.05,lambda:0.1,max_depth:3','eta:0.05,lambda:0.1,max_depth:10','eta:0.05,lambda:0.1,max_depth:20','eta:0.05,lambda:0.2,max_depth:3','eta:0.05,lambda:0.2,max_depth:10','eta:0.05,lambda:0.2,max_depth:20',
#'eta:0.1,lambda:0,max_depth:3','eta:0.1,lambda:0,max_depth:10','eta:0.1,lambda:0,max_depth:20','eta:0.1,lambda:0.1,max_depth:3','eta:0.1,lambda:0.1,max_depth:10','eta:0.1,lambda:0.1,max_depth:20','eta:0.1,lambda:0.2,max_depth:3','eta:0.1,lambda:0.2,max_depth:10','eta:0.1,lambda:0.2,max_depth:20',
#'eta:0.25,lambda:0,max_depth:3','eta:0.25,lambda:0,max_depth:10','eta:0.25,lambda:0,max_depth:20','eta:0.25,lambda:0.1,max_depth:3','eta:0.25,lambda:0.1,max_depth:10','eta:0.25,lambda:0.1,max_depth:20','eta:0.25,lambda:0.2,max_depth:3','eta:0.25,lambda:0.2,max_depth:10','eta:0.25,lambda:0.2,max_depth:20']
models    = None
classModel = None
#dataFrame = 'multiClassTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#NOTE random HP submission for nClass BDT
'''
script    = 'nClassBDT.py'
#param sets chosen randomly and submitted
paramSets = []

nIters = 1
nParams = 7

for i in range(nIters):
  paramSets.append('eta:%.2f' % np.random.uniform(0.05, 0.8))
  paramSets.append('gamma:%.2f' % np.random.uniform(0,2))
  paramSets.append('colsample_bytree:%.2f' % np.random.uniform(0.5,1))
  paramSets.append('max_depth:%g' % np.random.randint(3,15))  
  paramSets.append('subsample:%f' % np.random.uniform(0.8,1))  
  paramSets.append('min_child_weight:%g' % np.random.randint(2,15))  
  paramSets.append('lambda:%f' % np.random.uniform(0.1,2))

paramSets = [
  ",".join(paramSets[i:i+nParams])
  for i in xrange(0, len(paramSets), nParams)
]

models = None
classModel = None
#dataFrame = 'multiClassTotal.pkl'
dataFrame = None
sigFrame  = None

#NOTE: end of nClassBDT random HP opt
'''


#NOTE: nJet NN 
'''
script    = 'nJetNNCategorisation.py'
#paramSets =['hiddenLayers:2,nodes:300,dropout:0.1,batchSize:1000'] #best no weights model
#paramSets =['hiddenLayers:2,nodes:200,dropout:0.2,batchSize:1000'] #best cbrt(eq) weights model
#paramSets =['hiddenLayers:3,nodes:300,dropout:0.1,batchSize:1000'] #best sqrt(eq) weights model
#paramSets =['hiddenLayers:3,nodes:200,dropout:0.1,batchSize:1000'] #best eq weights model
paramSets =['hiddenLayers:3,nodes:300,dropout:0.1,batchSize:100'] #best eq weights model
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#NOTE: nJet NN optimisation (sweep through HPs grid) 
'''
script    = 'nJetNNCategorisation.py'
nParams = 4
paramSets =[]
#HP ranges:

hLayers       = [2,3,4]
nodesPerLayer = [100,200,300]
dropouts      = [0.1, 0.2]
batchSizes    = [100, 500, 1000]
#populate list
for layers in hLayers:
  for nodes in nodesPerLayer:
    for dropout in dropouts:
      for batchSize in batchSizes:
        paramSets.append('hiddenLayers:%i' % (layers))
        paramSets.append('nodes:%i' % (nodes))
        paramSets.append('dropout:%.1f' % (dropout))
        paramSets.append('batchSize:%i' % (batchSize))
paramSets = [
  ",".join(paramSets[i:i+nParams])
  for i in xrange(0, len(paramSets), nParams)
]

models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#NOTE: nClass NN 
'''
script    = 'nClassNN.py'
paramSets =['hiddenLayers:3,nodes:300,dropout:0.2,batchSize:500'] #best MC weights model
#paramSets =['hiddenLayers:3,nodes:200,dropout:0.2,batchSize:100'] #best EQ weights
#paramSets =['hiddenLayers:3,nodes:100,dropout:0.2,batchSize:100'] #best sqrt(eq) weights model
#paramSets =['hiddenLayers:4,nodes:300,dropout:0.2,batchSize:500'] #best cbrt(eq) model
#paramSets =['hiddenLayers:2,nodes:300,dropout:0.2,batchSize:1000'] #best no weights model
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#NOTE: nClass NN optimisation (sweep through HPs grid) 
'''
script    = 'nClassNN.py'
nParams = 4
paramSets =[]
#HP ranges:
hLayers       = [2,3,4]
nodesPerLayer = [100,200,300]
dropouts      = [0.1, 0.2]
batchSizes    = [100, 500, 1000]
#populate list
for layers in hLayers:
  for nodes in nodesPerLayer:
    for dropout in dropouts:
      for batchSize in batchSizes:
        paramSets.append('hiddenLayers:%i' % (layers))
        paramSets.append('nodes:%i' % (nodes))
        paramSets.append('dropout:%.1f' % (dropout))
        paramSets.append('batchSize:%i' % (batchSize))
paramSets = [
  ",".join(paramSets[i:i+nParams])
  for i in xrange(0, len(paramSets), nParams)
]

models    = None
classModel = None
#dataFrame = 'multiClassTotal.pkl'
dataFrame = None
sigFrame  = None
'''


#NOTE: Current implementation for ggH sigs with optional BDT

script    = 'dataSignificances.py'
models    = ['altDiphoModel__min_child_weight_5subsample_0.847204eta_0.75max_depth_7gamma_2.05.model'] 
paramSets = [None] # no effect here, just for submission to work
#classModel = None #reco only
#best nJet Models
#classModel = 'nJetModelWithMCWeights__min_child_weight_2__n_estimators_207__subsample_0.8920__eta_0.72__colsample_bytree_0.9__max_depth_14__gamma_0.01__lambda_0.9671.model' #best nJet W_mc model
#classModel = 'nJetModelWithEQWeights__min_child_weight_7__n_estimators_133__subsample_0.8151__eta_0.65__colsample_bytree_0.85__max_depth_8__gamma_0.08__lambda_0.1662.model' #best nJet W_eq model
#classModel = 'nJetModelWithSqrtEQWeights__min_child_weight_4__n_estimators_195__subsample_0.9518__eta_0.65__colsample_bytree_0.72__max_depth_14__gamma_0.01__lambda_0.4570.model' #best njet W_sqrtEW
#classModel = 'nJetModelWithCbrtEQWeights__min_child_weight_2__n_estimators_188__subsample_0.8661__eta_0.74__colsample_bytree_0.90__max_depth_9__gamma_0.04__lambda_0.3694.model' #best njet W_cbrtEW
#classModel = 'nJetModelWithNoWeights__min_child_weight_9__n_estimators_21__subsample_0.8943__eta_0.53__colsample_bytree_0.97__max_depth_7__gamma_2.00__lambda_0.7469.model' #best njet no weight model

#best nClass Models 
classModel = 'nClassesModelMCWeights___min_child_weight_13__n_estimators_18__sub_sample_0.9030__eta_0.76__colsample_bytree_0.95__max_depth_12__gamma_1.85__lambda_1.8283.model' #best nClass W_mc model
#classModel = 'nClassesModelEQWeights___min_child_weight_2__n_estimators_47__sub_sample_0.9614__eta_0.74__colsample_bytree_0.98__max_depth_6__gamma_0.03__lambda_1.1377.model' #best nClass W_EQ model
#classModel = 'nClassesModelSqrtEQWeights___min_child_weight_2__n_estimators_196__sub_sample_0.8505__eta_0.75__colsample_bytree_0.90__max_depth_9__gamma_0.08__lambda_0.9049.model' #best nClass Sqrt(EQ) model
#classModel = 'nClassesModelCbrtEQWeights___min_child_weight_2__n_estimators_245__sub_sample_0.8985__eta_0.63__colsample_bytree_0.99__max_depth_14__gamma_0.06__lambda_0.9890.model' #best nClass Cbrt(EQ) model
#classModel = 'nClassesModelNoWeights___min_child_weight_11__n_estimators_16__sub_sample_0.9683__eta_0.69__colsample_bytree_0.97__max_depth_8__gamma_0.91__lambda_0.2412.model' #best nClass No weights model
#classModel = 'nClassesModelTestRobustness___min_child_weight_13__n_estimators_18__sub_sample_0.9030__eta_0.76__colsample_bytree_0.95__max_depth_12__gamma_1.85__lambda_1.8283.model' #testing robustness
#classModel = 'nClassesModelTestPeturbedHPs___min_child_weight_12__n_estimators_17__sub_sample_0.9030__eta_0.7__colsample_bytree_0.95__max_depth_13__gamma_1.95__lambda_1.80.model' #testing again

for params in paramSets:
  if not params: continue
  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotal.pkl'
#dataFrame = None
#sigFrame  = 'signifTotal.pkl'
sigFrame  = 'MultiClassTotal.pkl'
#sigFrame  = None
########intLumi = 137. #NOTE: uncomment iff doing combined optimisation


#NOTE: Current implementation for ggH sigs with optional NN
'''
script    = 'dataSignificancesNN.py'
models    = ['altDiphoModel__min_child_weight_5subsample_0.847204eta_0.75max_depth_7gamma_2.05.model'] #optmimise this model too  #,'diphoModel.model']
paramSets = [None] # dipho model HPs
#classModel = None #reco
#Best nJet NN models
#classModel = 'nJetNN_MCWeights___hiddenLayers_3__nodes_300__dropout_0.1__batchSize_100.h5' #best nJet NN with MC weights
#classModel = 'nJetNN_EQWeights___hiddenLayers_3__nodes_200__dropout_0.1__batchSize_1000.h5' #best nJet NN with equal weights
#classModel = 'nJetNN_SqrtEQWeights___hiddenLayers_3__nodes_300__dropout_0.1__batchSize_1000.h5' #best nJet NN with sqrt equal weights
#classModel = 'nJetNN_CbrtEQWeights___hiddenLayers_2__nodes_200__dropout_0.2__batchSize_1000.h5' #best nJet NN with cbrt equal weights
#classModel = 'nJetNN_NoWeights___hiddenLayers_2__nodes_300__dropout_0.1__batchSize_1000.h5' #best nJet NN with no weights

#best nClass NN models
classModel = 'nClassesNNMCweights____hiddenLayers_3__nodes_300__dropout_0.2__batchSize_500.h5' #best nClass NN with MC weights
#classModel = 'nClassesNNEQweights____hiddenLayers_3__nodes_200__dropout_0.2__batchSize_100.h5' #best nClass NN with equal weights
#classModel = 'nClassesNNSqrtEQweights____hiddenLayers_3__nodes_100__dropout_0.2__batchSize_100.h5' #best nClass NN with sqrt equal weights
#classModel = 'nClassesNNCbrtEQweights____hiddenLayers_4__nodes_300__dropout_0.2__batchSize_500.h5' #best nClass NN with cbrt equal weights
#classModel = 'nClassesNNNOweights____hiddenLayers_2__nodes_300__dropout_0.2__batchSize_1000.h5' #best nJet NN with no weights

for params in paramSets:
  if not params: continue
  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotalNN.pkl'
#dataFrame = None
sigFrame  = 'signifTotalNN.pkl'
#sigFrame  = None
########intLumi = 137. #NOTE: uncomment iff doing combined optimisation
'''

#NOTE: Check NN assigned bkg distributions for cats and associated BG
'''
script    = 'bkgCheckDataSignificancesNN.py'
models    = ['altDiphoModel__min_child_weight_5subsample_0.847204eta_0.75max_depth_7gamma_2.05.model'] 
paramSets = [None] # no effect here, just for submission to work
#classModel = 'nJetNN_MCWeights___hiddenLayers_3__nodes_300__dropout_0.1__batchSize_100.h5'
classModel = 'nJetNN_EQWeights___hiddenLayers_3__nodes_200__dropout_0.1__batchSize_1000.h5'
for params in paramSets:
  if not params: continue
  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotalNN.pkl'
#dataFrame = None
sigFrame  = 'signifTotalNN.pkl'
#sigFrame  = None
'''
#NOTE: Check BDT assigned bkg distributions for cats and associated BG

'''
script    = 'bkgCheckDataSignificancesnJetBDT.py'
models    = ['altDiphoModel__min_child_weight_5subsample_0.847204eta_0.75max_depth_7gamma_2.05.model'] 
paramSets = [None] # no effect here, just for submission to work
classModel = 'nJetModelWithEQWeights__min_child_weight_7__n_estimators_133__subsample_0.8151__eta_0.65__colsample_bytree_0.85__max_depth_8__gamma_0.08__lambda_0.1662.model' #best nJet W_eq model
for params in paramSets:
  if not params: continue
  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotal.pkl'
#dataFrame = None
sigFrame  = 'signifTotal.pkl'
#sigFrame  = None
'''

#script    = 'dataMCcheckSidebands.py'
#models    = ['altDiphoModel.model','diphoModel.model']
#classModel = None
#paramSets = None
#dataFrame = 'dataTotal.pkl'
#sigFrame  = 'trainTotal.pkl'

#script    = 'dataSignificancesVBF.py'
##models    = [None,'altDiphoModel.model','diphoModel.model']
#models    = ['altDiphoModel.model','diphoModel.model']
#classModel = None
##paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
#paramSets = [None]
#for params in paramSets:
#  if not params: continue
#  params = params.split(',')
#  name = 'diphoModel'
#  for param in params:
#    var = param.split(':')[0]
#    val = param.split(':')[1]
#    name += '__%s_%s'%(var,str(val))
#  name += '.model'
#  models.append(name)
#  models.append(name.replace('dipho','altDipho'))
#paramSets = None
#dataFrame = None
##dataFrame = 'dataTotal.pkl'
#sigFrame  = None
##sigFrame  = 'vbfTotal.pkl'

#script    = 'combinedBDT.py'
#paramSets = None
#models    = [None,'altDiphoModel.model']
#classModel = None
##dataFrame = None
#dataFrame = 'combinedTotal.pkl'
#sigFrame  = None

#script    = 'dataSignificancesVBFcombined.py'
#models = [None,'altDiphoModel.model']
#classModel = None
#paramSets = None
##dataFrame = None
#dataFrame = 'dataTotal.pkl'
##sigFrame  = None
#sigFrame  = 'vbfTotal.pkl'

if __name__=='__main__':
  for year in years:
    jobDir = '%s/Jobs/%s/%s' % (myDir, script.replace('.py',''), year)
    if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
    trainDir  = '%s/%s/trees'%(baseDir,year)
    theCmd = 'python %s -t %s '%(script, trainDir)
    if dataFrame: 
      theCmd += '-d %s '%(dataFrame)
    if sigFrame: 
      theCmd += '-s %s '%sigFrame
    if intLumi: 
      theCmd += '--intLumi %s '%intLumi
    if classModel: 
      theCmd += '--className %s '%classModel
    if paramSets and models:
      exit('ERROR do not expect both parameter set options and models. Exiting..')
    elif paramSets: 
      for params in paramSets:
         
        fullCmd = theCmd 
        if params: fullCmd += '--trainParams %s '%params
        if not runLocal: submitJob( jobDir, fullCmd, params=params, dryRun=dryRun )
        elif dryRun: print fullCmd 
        else:
          print fullCmd
          system(fullCmd)
    elif models:
      for model in models:
        print(model)
        fullCmd = theCmd
        if model: fullCmd += '-m %s '%model
        if not runLocal: submitJob( jobDir, fullCmd, model=model, dryRun=dryRun )
        elif dryRun: print fullCmd
        else:
          print fullCmd
          system(fullCmd)
