#!/usr/bin/env python
from os import system, path, getcwd
from otherHelpers import submitJob
import numpy as np
from collections import OrderedDict as od

dryRun = False
#dryRun = True

#runLocal = False
runLocal = True

myDir = getcwd()
baseDir = '/vols/cms/jwd18/Stage1categorisation/Pass1'
years = od()
#years['2016'] = 35.9
#years['2017'] = 41.5
#years['2018'] = 59.7
years['Combined']  = 45.7


'''
script    = 'diphotonCategorisation.py'
paramSets = [None]
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
models    = None
classModel = None
#dataFrame = 'trainTotal.pkl'
dataFrame = None
sigFrame  = None
'''

#script    = 'vhHadCategorisation.py'
#paramSets = [None]
##paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
#models    = None
#classModel = None
##dataFrame = 'vhHadTotal.pkl'
#dataFrame = None
#sigFrame  = None

#script    = 'dataSignificancesVHhad.py'
#models    = ['altDiphoModel.model']
#classModel = None
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
#paramSets = None
#dataFrame = None
##dataFrame = 'dataTotal.pkl'
#sigFrame  = 'vhHadTotal.pkl'

'''
script    = 'nJetBDT.py'
paramSets =[None] 
#paramSets =['min_child_weight:10,subsample:0.949832,eta:0.76,colsample_bytree:0.95,max_depth:12,gamma:0.51,lambda:0.777625']  #best model for MC weights
#paramSets =['min_child_weight:3,subsample:0.847088,eta:0.68,colsample_bytree:0.86,max_depth:9,gamma:0.02,lambda:0.494380']  #best model EQ weights
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
script    = 'nJetBDTFullFeatures.py'
paramSets =[None] # best for equal classes
#paramSets =['min_child_weight:10,eta:0.76,max_depth:11,gamma:0.48,lambda:1.401161']  #best model for MC weights X 1000
paramSets =['min_child_weight:2,eta:0.75,max_depth:13,gamma:0.02,lambda:1.346733']  #best model for MC weights X 1000
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
script    = 'nJetBDTFullFeatures.py'
#param sets chosen randomly and submitted
paramSets = []

nIters = 4000
nParams = 5

for i in range(nIters):
  paramSets.append('eta:%.2f' % np.random.uniform(0.05, 0.8))
  paramSets.append('gamma:%.2f' % np.random.uniform(0,2))
  paramSets.append('max_depth:%g' % np.random.randint(3,15))  
  paramSets.append('min_child_weight:%g' % np.random.randint(2,15))  
  paramSets.append('lambda:%f' % np.random.uniform(0.1,2))

paramSets = [
  ",".join(paramSets[i:i+nParams])
  for i in xrange(0, len(paramSets), nParams)
]

models = None
classModel = None
dataFrame = None
sigFrame  = None
'''

'''
#NOTE: here onwards is random HP search for nJet
script    = 'nJetBDT.py'
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
'''

'''
script    = 'nClassBDT.py'
paramSets = [None]
#paramSets = ['min_child_weight:6,subsample:0.845330,eta:0.68,colsample_bytree:0.99,max_depth:13,gamma:0.41,lambda:1.458330'] # MCW, no dipho pt
#paramSets = ['min_child_weight:3,subsample:0.847088,eta:0.68,colsample_bytree:0.86,max_depth:9,gamma:0.02,lambda:0.494380'] # EQW, no dipho pt
#paramSets = ['min_child_weight:3,subsample:0.843205,eta:0.72,colsample_bytree:0.95,max_depth:6,gamma:0.11,lambda:0.919859'] # MCW, dipho pt
#paramSets = ['min_child_weight:2,subsample:0.964181,eta:0.72,colsample_bytree:0.86,max_depth:9,gamma:0.07,lambda:1.002382'] # EQW, dipho pt
models    = None
classModel = None
#dataFrame ='MultiClassTotal.pkl'
dataFrame = None
sigFrame = None
'''

'''
script    = 'nClassBDT.py'
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
#dataFrame = 'multiClassTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
script    = 'EightClassBDT.py'
paramSets = [None]
#paramSets = ['min_child_weight:11,subsample:0.976559,eta:0.78,colsample_bytree:0.89,max_depth:13,gamma:0.65,lambda:1.615699'] # MCW, no dipho pt
paramSets = ['min_child_weight:2,subsample:0.942551,eta:0.57,colsample_bytree:0.80,max_depth:6,gamma:0.03,lambda:0.327904'] # EQW, no dipho pt
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame = None
'''

'''
script    = 'EightClassBDT.py'
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
#dataFrame = 'multiClassTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
script    = 'nClassNN.py'
paramSets =['hiddenLayers:2,nodes:100,dropout:0.1,batchSize:500'] #best MC weights model
#paramSets =[None] #best eq weights model
models    = None
classModel = None
#dataFrame = 'nClassNNTotal.pkl'
dataFrame = None
sigFrame  = None
'''

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

'''
script    = 'nJetNN.py'
paramSets =['hiddenLayers:2,nodes:100,dropout:0.1,batchSize:500'] #best MC weights model
models    = None
classModel = None
#dataFrame = 'jetTotal.pkl'
dataFrame = None
sigFrame  = None
'''

'''
script    = 'nJetNN.py'
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


script    = 'dataSignificances.py'
#models    = ['altDiphoModelCombined.model']
models    = ['altDiphoModel_NewModel.model']
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
paramSets = [None]
classModel = None
classModel = 'nClassesModelMCW_Combined_.model' #combined
#classModel = 'nClassesModelMCW___min_child_weight_6__subsample_0.845330__eta_0.68__colsample_bytree_0.99__max_depth_13__gamma_0.41__lambda_1.458330.model'#best nClass model trained with MC weights
#classModel = 'nJetModel_MCW___min_child_weight_10__subsample_0.949832__eta_0.76__colsample_bytree_0.95__max_depth_12__gamma_0.51__lambda_0.777625.model' #best nJet model trained with MC weights
#classModel = 'nClassesModelEQW___min_child_weight_2__subsample_0.96__eta_0.68__colsample_bytree_0.86__max_depth_9__gamma_0.02__lambda_0.494380.model' #best nlass model trained with EQ weights (doesnt exist rn)
#classModel = 'nJetModel_EQW___min_child_weight_3__subsample_0.847088__eta_0.68__colsample_bytree_0.86__max_depth_9__gamma_0.02__lambda_0.494380.model' #best nJet model trained with EQ weights

#classModel = 'nClassesModelMCW_withPt___min_child_weight_3__subsample_0.843205__eta_0.72__colsample_bytree_0.95__max_depth_6__gamma_0.11__lambda_0.919859.model' #best nClass model trained with MC weights and dipho_pt
#classModel = 'nClassesModelEQW_withPt___min_child_weight_2__subsample_0.964181__eta_0.72__colsample_bytree_0.86__max_depth_9__gamma_0.07__lambda_1.002382.model' #best nClass model trained with EQ weights and dipho_pt
for params in paramSets:
  if not params: continue
#  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotal.h5'
dataFrame = None
sigFrame  = 'signifTotal.h5'
sigFrame  = None


'''
script    = 'dataSignificancesEightCat.py'
models    = ['altDiphoModel.model']
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
paramSets = [None]

#script    = 'vbfCategorisation.py'
#paramSets = [None]
##paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
#models    = None
#classModel = None
##dataFrame = 'vbfTotal.pkl'
#dataFrame = None
#sigFrame  = None

#script    = 'dataSignificancesVBFthree.py'
#models    = ['altDiphoModel.model']
#classModel = None
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
#paramSets = None
#dataFrame = None
##dataFrame = 'dataTotal.pkl'
#sigFrame  = 'vbfTotal.pkl'

#script    = 'nJetCategorisation.py'
#paramSets = [None,'max_depth:10']
#models    = None
#classModel = None
classModel = 'EightClassesModelMCW___min_child_weight_11__subsample_0.976559__eta_0.78__colsample_bytree_0.89__max_depth_13__gamma_0.65__lambda_1.615699.model' #best EightClass model trained with MC weights and dipho_pt
#classModel = 'EightClassesModelEQW___min_child_weight_2__subsample_0.942551__eta_0.57__colsample_bytree_0.80__max_depth_6__gamma_0.03__lambda_0.327904.model' #best EightClass model trained with EQW weights and dipho_pt
for params in paramSets:
  if not params: continue
#  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
#dataFrame = 'dataTotal.pkl'
dataFrame = None
#sigFrame  = 'signifTotal.pkl'
sigFrame  = None
'''

'''
script    = 'dataSignificancesNN.py'
models    = ['altDiphoModel.model']
#classModel = None #reco
paramSets = [None]
#classModel = 'nJetNN_MCWeights___hiddenLayers_2__nodes_100__dropout_0.1__batchSize_500.h5'
classModel = 'nClassesNN_MCweights____hiddenLayers_2__nodes_100__dropout_0.1__batchSize_500.h5'
#classModel = None
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
#sigFrame  = 'nClassNNTotal.pkl'
sigFrame  = None
'''

'''
script    = 'dataSignificancesNjetFullFeatures.py'
models    = ['altDiphoModel.model']
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
paramSets = [None]
classModel = None
#classModel = 'nJet_FullFeatures.model' #MCW
classModel = 'nJetModel_FullFeatures_EQW___min_child_weight_2__eta_0.75__max_depth_13__gamma_0.02__lambda_1.346733.model' #EQW
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
#dataFrame = 'dataTotal.pkl'
dataFrame = None
#sigFrame  = 'signifTotal.pkl'
sigFrame  = None
'''

#script    = 'dataSignificances_2016_data.py'
#models    = ['altDiphoModel__min_child_weight_5subsample_0.847204eta_0.75max_depth_7gamma_2.05.model']
##paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
#paramSets = [None]
#classModel = None
##classModel = 'jetModel.model'
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
##dataFrame = 'dataTotal.pkl'
#dataFrame = None
##sigFrame  = 'signifTotal.pkl'
#sigFrame  = None


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
  #for year in years:
  for year,lumi in years.iteritems():
    jobDir = '%s/Jobs/%s/%s' % (myDir, script.replace('.py',''), year)
    if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
    trainDir  = '%s/%s/trees/'%(baseDir,year)
    theCmd = 'python %s -t %s '%(script, trainDir)
    theCmd += '--intLumi %s '%lumi
    if dataFrame: 
      theCmd += '-d %s '%dataFrame
    if sigFrame: 
      theCmd += '-s %s '%sigFrame
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

