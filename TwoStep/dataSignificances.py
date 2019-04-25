#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthClass
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma, jetPtToClass
from root_numpy import tree2array, fill_hist
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signifFrame', default=None, help='Name of cleaned signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('-c','--className', default=None, help='Name of multi-class model used to build categories. If None, use reco categories')
parser.add_option('-n','--nIterations', default=1, help='Number of iterations to run for random significance optimisation') #previously was default = 2000
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity') #def:35.9
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')
plotDir  = trainDir.replace('trees','plots')
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)
nJetClasses = 3
nClasses = 9 #runs 0->8 in opt loop 
jetPriors = [0.606560, 0.270464, 0.122976]
lumiScale = 137./35.9
#put root in batch mode
r.gROOT.SetBatch(True)
#bool to chose to scale to lumi other than 35.9
scaleToLumi = False

#get trees from files, put them in data frames
procFileMap = {'Data':'Data.root'}
theProcs = procFileMap.keys()

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

jetVars  = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dZ',0)
      trainTree.SetBranchStatus('centralObjectWeight',0)
      trainTree.SetBranchStatus('rho',0)
      trainTree.SetBranchStatus('event',0)
      trainTree.SetBranchStatus('lumi',0)
      trainTree.SetBranchStatus('processIndex',0)
      trainTree.SetBranchStatus('run',0)
      trainTree.SetBranchStatus('npu',0)
      trainTree.SetBranchStatus('puweight',0)
      newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
      newTree = trainTree.CloneTree()
      trainFrames[proc] = pd.DataFrame( tree2array(newTree) )
      del newTree
      del newFile
      trainFrames[proc]['proc'] = proc #adds process label to each event. i.e. if we read in ggH, proc = ggH
  print 'got trees'
 
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  dataTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass>100.]
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  dataTotal = dataTotal[dataTotal.leadmva>-0.9]
  dataTotal = dataTotal[dataTotal.subleadmva>-0.9]
  dataTotal = dataTotal[dataTotal.leadptom>0.333]
  dataTotal = dataTotal[dataTotal.subleadptom>0.25]
  print 'done basic preselection cuts'
  
  #Change the backgrounf weights if we want to scale to higher/lower lumi:
  if(scaleToLumi==True):
    print('Scaling background to alternative lumi. First weights before scaling:')
    print(dataTotal['weights'].head(5))
    dataTotal['weight'] = dataTotal['weight']*lumiScale
    print('Weights after scaling:')
    print(dataTotal['weights'].head(5))

  #add extra info to dataframe
  print 'about to add extra columns'
  dataTotal['diphopt'] = dataTotal.apply(addPt, axis=1)
  dataTotal['reco'] = dataTotal.apply(reco, axis=1)
  print 'all columns added'
  print 'first few columns in dataTotal are:'
  print(dataTotal.head())

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

if not opts.signifFrame:
  #sigFileMap = {'ggh':'ggH.root'}
  sigFileMap = {'ggh':'Merged.root'}
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in sigFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc:
        trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else:
        trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dijet_*',0)
      trainTree.SetBranchStatus('dijet_Mjj',1) #Need this on for 1.1
      trainTree.SetBranchStatus('dZ',0)
      trainTree.SetBranchStatus('centralObjectWeight',0)
      trainTree.SetBranchStatus('rho',0)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('event',0)
      trainTree.SetBranchStatus('lumi',0)
      trainTree.SetBranchStatus('processIndex',0)
      trainTree.SetBranchStatus('run',0)
      trainTree.SetBranchStatus('npu',0)
      trainTree.SetBranchStatus('puweight',0)
      newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
      newTree = trainTree.CloneTree()
      trainFrames[proc] = pd.DataFrame( tree2array(newTree) )
      del newTree # remove this if going to use the tree for a 2D histo 
      del newFile
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc,fn in sigFileMap.iteritems():
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.stage1cat>-1.]

  #DEBUG
  print('features of the signal dataframe are:')
  print(list(trainTotal.columns))

  #read in signal mc dataframe
  # trainTotal = pd.read_pickle('%s/trainTotal.pkl'%frameDir)
  #trainTotal = pd.read_pickle('%s/jetTotal.pkl'%frameDir)
  print 'Successfully loaded the signal dataframe'
  
  #remove bkg then add reco tag info
  #trainTotal = trainTotal[trainTotal.stage1cat>0.01]
  #trainTotal = trainTotal[trainTotal.stage1cat<12.]
  #trainTotal = trainTotal[trainTotal.stage1cat!=1]
  #trainTotal = trainTotal[trainTotal.stage1cat!=2]

  # updated to reco for 1.1 and remove bin -1 (filled when no other bins satisfied)
  print 'About to add reco tag info'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho, axis=1)
  trainTotal['truthClass'] = trainTotal.apply(truthClass, axis=1)
  print 'Successfully added reco tag info'

  #save
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/signifTotal.pkl'%frameDir)
  print 'MC signal frame saved as %s/signifTotal.pkl'%frameDir

else:
  #read in already cleaned up signal frame
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signifFrame))


#define the variables used as input to the classifier
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
#diphoI  = trainTotal[jetVars].values
diphoJ  = trainTotal['truthClass'].values
diphoFW = trainTotal['weight'].values
#diphoS  = trainTotal['stage1cat'].values
diphoP  = trainTotal['diphopt'].values
diphoR  = trainTotal['reco'].values
diphoM  = trainTotal['CMS_hgg_mass'].values

dataX  = dataTotal[diphoVars].values
#dataI  = dataTotal[jetVars].values
dataY  = np.zeros(dataX.shape[0])
dataFW = np.ones(dataX.shape[0])
dataP  = dataTotal['diphopt'].values
dataR  = dataTotal['reco'].values
dataM  = dataTotal['CMS_hgg_mass'].values


#setup matrices
diphoMatrix = xg.DMatrix(diphoX, label=diphoY, weight=diphoFW, feature_names=diphoVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataY,  weight=dataFW,  feature_names=diphoVars)

#load the dipho model to be tested
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))

#get predicted values (MVA values)
diphoPredY = diphoModel.predict(diphoMatrix)
dataPredY  = diphoModel.predict(dataMatrix)

#load the class model to be tested, if it exists
if opts.className:
  classModel = xg.Booster()
  classModel.load_model('%s/%s'%(modelDir,opts.className))

  if 'jet' in opts.className:
    classMatrix = xg.DMatrix(diphoI, label=diphoY, weight=diphoFW, feature_names=jetVars)
    predProbJet = classModel.predict(classMatrix).reshape(diphoX.shape[0],nJetClasses)
    predProbJet *= jetPriors
    classPredJ = np.argmax(predProbJet, axis=1)
    diphoR = jetPtToClass(classPredJ, diphoP)

    classDataMatrix = xg.DMatrix(dataI, label=dataY, weight=dataFW, feature_names=jetVars)
    predProbJet = classModel.predict(classDataMatrix).reshape(dataX.shape[0],nJetClasses)
    predProbJet *= jetPriors
    classPredJ = np.argmax(predProbJet, axis=1)
    dataR = jetPtToClass(classPredJ, dataP)
  else:
    raise Exception("your class model type is not yet supported, sorry")

#now estimate two-class significance
#set up parameters for the optimiser
ranges = [ [0.5,1.] ]
names  = ['DiphotonBDT']
printStr = ''
BDTStr = ''
SigList = []
NSignalEvents = []
NBackgroundEvents = []

plotDir  = '%s/%s/Proc_0'%(plotDir,opts.modelName.replace('.model',''))
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)


for iClass in range(nClasses):
  sigWeights = diphoFW * (diphoJ==iClass) * (diphoR==iClass)
  NSignalEvents.append(sigWeights.sum()*35.9)
  bkgWeights = dataFW * (dataR==iClass)
  NBackgroundEvents.append(bkgWeights.sum())
  optimiser = CatOptim(sigWeights, diphoM, [diphoPredY], bkgWeights, dataM, [dataPredY], 2, ranges, names)
  #optimiser.setTransform(True) #FIXME
  optimiser.optimise(opts.intLumi, opts.nIterations)
  plotDir  = plotDir.replace('Proc_%g'%(iClass-1),'Proc_%g'%iClass)
  if not path.isdir(plotDir): 
    system('mkdir -p %s'%plotDir)
  #optimiser.crossCheck(opts.intLumi,plotDir)
  printStr += 'Results for bin %g : \n'%iClass
  printStr += optimiser.getPrintableResult()
  toPrint = optimiser.getPrintableResult()
  splits = toPrint.split(':')
  BDTStr += (splits[1][14:19])
  BDTStr += '\n'
  BDTStr += (splits[2][14:19])
  BDTStr += '\n'
  Sig = float(splits[2][-8:-3])
  
  #NOTE: scale sig by sqrt of lumi scale if doing opt by indiv years
  SigList.append(round(Sig,2))

#No bins requiring 3 cats for 1.1
#binsRequiringThree = [0] 
#for iClass in binsRequiringThree:
  #sigWeights = diphoFW * (diphoJ==iClass) * (diphoR==iClass)
  #bkgWeights = dataFW * (dataR==iClass)
  #optimiser = CatOptim(sigWeights, diphoM, [diphoPredY], bkgWeights, dataM, [dataPredY], 3, ranges, names)
  #optimiser.setTransform(True) #FIXME
  #optimiser.optimise(opts.intLumi, opts.nIterations)
  #printStr += 'Results for bin %g : \n'%iClass
  #printStr += optimiser.getPrintableResult()
'''
BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/BDTCutsLumi137.txt','w+')
BDTCutFile.write('%s'%BDTStr)
BDTCutFile.close()
CatSigsFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/CatSigsLumi137.txt','w+')
CatSigsFile.write('%s'%SigList)
CatSigsFile.close()
'''
print(NSignalEvents)
print(NBackgroundEvents)

print
print printStr
