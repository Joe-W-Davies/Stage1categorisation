#!/usr/bin/env python
from os import system, path, getcwd
from otherHelpers import submitJob

#Set false if not wanting to submit to batch
dryRun = False
#dryRun = True

myDir = getcwd()
# change the ggH.root MC file in /Pass1 to the merged root file from lxplus, that has gen level info in 
# run flashGG with the data files too... why?
baseDir = '/vols/cms/jwd18/Stage1categorisation/Pass1'
#years = ['2016','2017']

years = ['2016']
intLumi = 35.9

#years = ['2017']
#intLumi = 41.5

#script    = 'diphotonCategorisation.py'

#We are going to test 30 scenarious for three hyperparameters. 
#Each will have a "low", "medium", and "high" value, leading to
# 29 combinations, plus a test scenario with default values.
# eta = 0.05, 0.1, 0.25
# lambda = 0, 0.1, 0.2 
# max_depth = 3,10,20

#paramSets = ['eta:0.25,lambda:0.3,max_depth:8']

#paramSets = ['eta:0.05,lambda:0,max_depth:3','eta:0.05,lambda:0,max_depth:10','eta:0.05,lambda:0,max_depth:20','eta:0.05,lambda:0.1,max_depth:3','eta:0.05,lambda:0.1,max_depth:10','eta:0.05,lambda:0.1,max_depth:20','eta:0.05,lambda:0.2,max_depth:3','eta:0.05,lambda:0.2,max_depth:10','eta:0.05,lambda:0.2,max_depth:20',
#'eta:0.1,lambda:0,max_depth:3','eta:0.1,lambda:0,max_depth:10','eta:0.1,lambda:0,max_depth:20','eta:0.1,lambda:0.1,max_depth:3','eta:0.1,lambda:0.1,max_depth:10','eta:0.1,lambda:0.1,max_depth:20','eta:0.1,lambda:0.2,max_depth:3','eta:0.1,lambda:0.2,max_depth:10','eta:0.1,lambda:0.2,max_depth:20',
#'eta:0.25,lambda:0,max_depth:3','eta:0.25,lambda:0,max_depth:10','eta:0.25,lambda:0,max_depth:20','eta:0.25,lambda:0.1,max_depth:3','eta:0.25,lambda:0.1,max_depth:10','eta:0.25,lambda:0.1,max_depth:20','eta:0.25,lambda:0.2,max_depth:3','eta:0.25,lambda:0.2,max_depth:10','eta:0.25,lambda:0.2,max_depth:20']
#models = None
#classModel = None
#dataFrame = 'trainTotal.pkl'
#dataFrame = None
#sigFrame  = None

#Create dict of hyperparams and values for use in BayesCV
#paramSets = {
       # 'learning_rate': (0.01, 1.0, 'log-uniform'),
       # 'min_child_weight': (0, 10),
       # 'max_depth': (0, 50),
       # 'max_delta_step': (0, 20),
       # 'subsample': (0.01, 1.0, 'uniform'),
       # 'colsample_bytree': (0.01, 1.0, 'uniform'),
       # 'colsample_bylevel': (0.01, 1.0, 'uniform'),
       # 'reg_lambda': (1e-9, 1000, 'log-uniform'),
       # 'reg_alpha': (1e-9, 1.0, 'log-uniform'),
       # 'gamma': (1e-9, 0.5, 'log-uniform'),
       # 'min_child_weight': (0, 5),
       # 'n_estimators': (50, 100),
       # 'scale_pos_weight': (1e-6, 500, 'log-uniform')
    #}

#script    = 'nJetCategorisation.py'
#paramSets = [None,'max_depth:10']
#models    = None
#classModel = None
##dataFrame = 'jetTotal.pkl'
#dataFrame = None
#sigFrame  = None

script    = 'dataSignificances.py'
models    = ['altDiphoModel.model'] #,'diphoModel.model']
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
paramSets = [None]
classModel = None
#classModel = 'jetModel.model'
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
sigFrame  = None

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
      theCmd += '-d %s '%dataFrame
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
        submitJob( jobDir, fullCmd, params=params, dryRun=dryRun )
    elif models:
      for model in models:
        print(model)
        fullCmd = theCmd
        if model: fullCmd += '-m %s '%model
        submitJob( jobDir, fullCmd, model=model, dryRun=dryRun )
