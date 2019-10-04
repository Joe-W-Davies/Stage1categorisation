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
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from os import path, system
from array import array
from math import pi

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthClass, jetPtToggHClass
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma 
from root_numpy import tree2array, fill_hist
from catOptim import CatOptim
import usefulStyle as useSty

from keras.models import load_model
import h5py

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
catNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high', 'BSM']
binNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high', 'BSM', 'VBF-like']
jetPriors = [0.606560, 0.270464, 0.122976]
procPriors= [0.130697, 0.478718, 0.149646, 0.098105, 0.018315, 0.028550, 0.040312, 0.027654, 0.028003]
#lumiScale = 137./35.9
#put root in batch mode
r.gROOT.SetBatch(True)
#bool to chose to scale to lumi other than 35.9
scaleToLumi = False
#For adding priors if using ML categorisation
addPriors = False

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

allVars   = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

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
  #remove STXS bins > 9 (at reco level! can still have gen contamination)
  dataTotal = dataTotal[dataTotal.dijet_Mjj<350]
  print 'done basic preselection cuts'
  
  #Change the backgroung weights if we want to scale to higher/lower lumi:
  if(scaleToLumi==True):
    print('Scaling background to alternative lumi. First weights before scaling:')
    print(dataTotal['weight'].head(5))
    dataTotal['weight'] = dataTotal['weight']*lumiScale
    print('Weights after scaling:')
    print(dataTotal['weight'].head(5))

  #add extra info to dataframe
  print 'about to add extra columns'

  dataTotal['diphopt'] = dataTotal.apply(addPt, axis=1)
  dataTotal['reco'] = dataTotal.apply(reco, axis=1)
  print 'all columns added'
  #print 'first few columns in dataTotal are:'
  #print(dataTotal.head())

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

# Do same thing for the MC signal dataframe
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
      #trainTree.SetBranchStatus('dijet_*',0) #Need di-jet if using jet classifier model
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
  #remove STXS bins > 9 at reco level! can still have gen contamination
  trainTotal = trainTotal[trainTotal.dijet_Mjj<350] 
  
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
  #FIXME: temporary debug
  trainTotal = trainTotal[trainTotal.truthClass<9]
  print 'Successfully added reco tag info'

  #save
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/signifTotal.pkl'%frameDir)
  print 'MC signal frame saved as %s/signifTotal.pkl'%frameDir

else:
  #read in already cleaned up signal frame
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signifFrame))
  print 'Successfully loaded the signal dataframe'

#define the variables used as input to the classifier
if (opts.className):
  if 'Jet' in opts.className:
    diphoI  = trainTotal[jetVars].values 
  elif 'nClasses' in opts.className:
    diphoI  = trainTotal[allVars].values 
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
diphoJ  = trainTotal['truthClass'].values
diphoFW = trainTotal['weight'].values
diphoP  = trainTotal['diphopt'].values
diphoR  = trainTotal['reco'].values
diphoM  = trainTotal['CMS_hgg_mass'].values
diphoMVA= trainTotal['diphomvaxgb'].values

if (opts.className):
  if 'Jet' in opts.className:
    dataI  = dataTotal[jetVars].values 
  elif 'nClasses' in opts.className:
    dataI  = dataTotal[allVars].values 
dataX  = dataTotal[diphoVars].values
dataY  = np.zeros(dataX.shape[0])
dataFW = dataTotal['weight'].values
dataP  = dataTotal['diphopt'].values
dataR  = dataTotal['reco'].values
dataM  = dataTotal['CMS_hgg_mass'].values
dataMVA= dataTotal['diphomvaxgb'].values

#setup matrices for predicting dipho (BG rejection BDT)
diphoMatrix = xg.DMatrix(diphoX, label=diphoY, weight=diphoFW, feature_names=diphoVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataY,  weight=dataFW,  feature_names=diphoVars)

#load the dipho model to be tested
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))

#get predicted values from dipho model (1 or 0)
diphoPredY = diphoModel.predict(diphoMatrix)
dataPredY  = diphoModel.predict(dataMatrix)

print('reco accuracy score is:')
recoScore = accuracy_score(diphoJ, diphoR, sample_weight=diphoFW)
print(recoScore)
#load the classifier model to be tested, if it exists
#only trained the model to predict 9 categories, so exclude these from cats but keep in gen bins (purity mat)
if opts.className:
  #BDTs
  classModel = xg.Booster()
  classModel.load_model('%s/%s'%(modelDir,opts.className))
  
  if 'Jet' in opts.className:
    #predict class of signal
    classMatrix = xg.DMatrix(diphoI, label=diphoY, weight=diphoFW, feature_names=jetVars)
    predProbJet = classModel.predict(classMatrix).reshape(diphoX.shape[0],nJetClasses)
    #if addPriors: predProbJet *= jetPriors # assumes model trained with equal weights! 
    classPredJ = np.argmax(predProbJet, axis=1)
    #concat the nJet prediction with everything else you need to make class predction
    predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(diphoP)], axis=1)
    predFrame.columns = ['n_pred_jets', 'diphopt']
    predFrame['predClass'] = predFrame.apply(jetPtToggHClass, axis=1)
    diphoR = predFrame['predClass'].values  
    #save predicted signal classes for use in purity matrices
    print('nJet BDT accuracy score is:')
    print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))
    dfForDiphoBDTPurityMatrices = pd.concat([pd.DataFrame(diphoR), pd.DataFrame(diphoJ), pd.DataFrame(diphoFW), pd.DataFrame(diphoMVA)], axis=1)
    dfForDiphoBDTPurityMatrices.to_pickle('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoBDTCutsForMLCat/nJetBDT/BDTPredictedCategoriesSqrtEQWeights.pkl')
      
    #same thing for background
    classDataMatrix = xg.DMatrix(dataI, label=dataY, weight=dataFW, feature_names=jetVars)
    predProbJet = classModel.predict(classDataMatrix).reshape(dataX.shape[0],nJetClasses)
    if addPriors: predProbJet *= jetPriors 
    classPredJ = np.argmax(predProbJet, axis=1)
    predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(dataP)], axis=1)
    predFrame.columns = ['n_pred_jets', 'diphopt']
    predFrame['predClass'] = predFrame.apply(jetPtToggHClass, axis=1)
    dataR = predFrame['predClass'].values
 

  elif 'nClasses' in opts.className:
    classMatrix = xg.DMatrix(diphoI, label=diphoY, weight=diphoFW, feature_names=allVars)
    predProbJet = classModel.predict(classMatrix).reshape(diphoX.shape[0],nClasses)
    diphoR = np.argmax(predProbJet, axis=1)
    print('nClasses BDT accuracy score:')
    print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))
 
    #same thing for background (no need to save for purits matrices though)
    classDataMatrix = xg.DMatrix(dataI, label=dataY, weight=dataFW, feature_names=allVars)
    predProbJet = classModel.predict(classDataMatrix).reshape(dataX.shape[0],nClasses)
    dataR = np.argmax(predProbJet, axis=1)

  else:
    raise Exception("your class model type is not yet supported, sorry")

#now estimate two-class significance for the alt dipho model
#set up parameters for the optimiser
ranges = [ [0.5,1.] ]
names  = ['DiphotonBDT']
printStr = ''
BDTStr = ''
SigStr = '' 
sigList = []
NSignalEvents = []
NBackgroundEvents = []

plotDir  = '%s/%s/Proc_0'%(plotDir,opts.modelName.replace('.model',''))
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)


for iClass in range(nClasses):
  sigWeights = diphoFW * (diphoJ==iClass) * (diphoR==iClass)
  #NSignalEvents.append(sigWeights.sum()*35.9)
  bkgWeights = dataFW * (dataR==iClass)
  #NBackgroundEvents.append(bkgWeights.sum())
  optimiser = CatOptim(sigWeights, diphoM, [diphoPredY], bkgWeights, dataM, [dataPredY], 2, ranges, names)
  #optimiser.setTransform(True) #FIXME
  optimiser.optimise(opts.intLumi, opts.nIterations, iClass)
  #optimiser.crossCheck(opts.intLumi,plotDir)
  if(iClass!=5):
    printStr += 'Results for bin %g : \n'%iClass
    printStr += optimiser.getPrintableResult(iClass)
    toPrint = optimiser.getPrintableResult(iClass)
    splits = toPrint.split(':')
    BDTStr += (splits[1][14:19])
    BDTStr += '\n'
    BDTStr += (splits[2][14:19])
    BDTStr += '\n'
    Sig = float(splits[2][-9:-3])
 
    ''' 
    # For doing lumi scaling for combined and non combined. (Just care about 2016 only for now though)
    if(scaleToLumi == True):
      SigStr += '%f'%Sig
      SigStr += '\n'
    else:
      SigStr += '%f'%(Sig*(math.sqrt(lumiScale)))
      SigStr += '\n' 
     '''

    SigStr += '%f'%Sig
    SigStr += '\n'
    sigList.append(Sig)
  else:
    cat6Info = optimiser.getPrintableResult(iClass) 
    sigList.append(0)


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


#print(NSignalEvents)
#print(NBackgroundEvents)

print
print printStr

print('Accuracy is:')
print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))

#print the cuts to the files for use in analysis.py later (plotting confusion matrices using these cuts)
'''
#NOTE: These both print cuts/sigs for scaling to 137fb^-1. If you want it for 2016,
#NOTE: run on 36.9 without background scaling, and look at BDT cuts file (indiv). Sigs will
#NOTE: change since we multiply by sqrt(alpha), b ut BDT cuts do not, as found through earlier opt. 
if(scaleToLumi==True):
  BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/BDTCutsComb.txt','w+')
  BDTCutFile.write('%s'%BDTStr)
  BDTCutFile.close()
  CatSigsFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/CatSigsComb.txt','w+')
  CatSigsFile.write('%s'%SigStr)
  CatSigsFile.close()

else:
  BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/BDTCutsIndiv.txt','w+')
  BDTCutFile.write('%s'%BDTStr)
  BDTCutFile.close()
  CatSigsFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/CatSigsIndiv.txt','w+')
  CatSigsFile.write('%s'%SigStr)
  CatSigsFile.close()
'''



#for nJet BDT options. NOTE: change this for each weight scenario you are considering
'''
if opts.className:
  if 'Jet' in opts.className:
    BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/BDT/diphoBDTCutsWithNoWeights.txt','w+')
    BDTCutFile.write('%s'%BDTStr)
    BDTCutFile.close()
    sigsCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/BDT/catSigsWithNoWeights.txt','w+')
    sigsCutFile.write('%s'%printStr)
    sigsCutFile.close()

#same as above, but for the BDT multiclass model
  elif 'nClasses' in opts.className:
    BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nClassOutputs/BDT/diphoBDTCutsWithNoWeights.txt','w+')
    BDTCutFile.write('%s'%BDTStr)
    BDTCutFile.close()
    sigsCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nClassOutputs/BDT/catSigsNoWeights.txt','w+')
    sigsCutFile.write('%s'%printStr)
    sigsCutFile.close()

else: #reco
    BDTCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/diphoBDTCutsReco.txt','w+')
    BDTCutFile.write('%s'%BDTStr)
    BDTCutFile.close()
    sigsCutFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/catSigsReco.txt','w+')
    sigsCutFile.write('%s'%printStr)
    sigsCutFile.close()
'''

                                         ##### Plot cat6 Info #####
#Set up color pallete and other canvas options
npoints = 2
stops = [0.00,1.00]
red = [1.00, 0.50]
green = [1.00, 0.00]
blue = [1.00, 0.70]
ncontours = 256
alpha=1.

stops = array('d',stops)
red = array('d',red)
green = array('d',green)
blue = array('d',blue)

r.TColor.CreateGradientColorTable(npoints, stops, red, green, blue, ncontours, alpha)
r.gStyle.SetNumberContours(256)

                               ##### Plot cat 6 stuff #####
cat6HistDiphoCuts = r.TH2F('histo', ';;;Combined AMS value', 101, 0.5, 1.0, 101, 0.5, 1.0)
cut1, cut2 = zip(*cat6Info)
AMS = cat6Info.values()
for iCut1, iCut2, iAMS in zip(cut1, cut2, AMS):
  cat6HistDiphoCuts.Fill(iCut1, iCut2, iAMS)

#draw and save the 2D hists
canv = r.TCanvas()
canv.SetRightMargin(0.15)
canv.SetLeftMargin(0.12)
canv.SetBottomMargin(0.12)
canv.SetRightMargin(0.15)
cat6HistDiphoCuts.Draw('COLZ')
cat6HistDiphoCuts.GetXaxis().SetTitleOffset(1.4)
cat6HistDiphoCuts.GetXaxis().SetTitle('Diphoton BDT cut 1')
cat6HistDiphoCuts.GetYaxis().SetTitle('Diphoton BDT cut 2')
cat6HistDiphoCuts.GetYaxis().SetTitleOffset(1.4)
cat6HistDiphoCuts.GetZaxis().SetTitleOffset(1.2)
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='35.9 fb^{-1} (2016)')
#canv.Print("/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/catSixAMSTwoDimScan.pdf")


##### Plot purity matrices #####
#Here, we dont have reco cats for ggH vbf-like events, but we do get allow gen level contamination
#of the vbf-like events into the ggH reco categories


#NB diphoJ = Truth, diphoR = Cat from either reco or ML

#declare 2D hists
nBinsX=nClasses+1 #include the ggh VBF-like procs
nBinsY=nClasses 
procHistReco = r.TH2F('procHistReco',';;;Signal composiion (%)', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistReco.SetTitle('')
prettyHist(procHistReco)
procHistPred = r.TH2F('procHistPred','procHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistPred.SetTitle('')
prettyHist(procHistPred)
catHistReco  = r.TH2F('catHistReco',';;;Signal composition (%)', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistReco.SetTitle('')
prettyHist(catHistReco)
catHistPred  = r.TH2F('catHistPred','catHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistPred.SetTitle('')
prettyHist(catHistPred)

#generate weights for the 2D hists   

#lump all the VBF-like procs into one by at generator level, for aesthetic
diphoJ[diphoJ > 8] = 9

#Sum weights for each bin i.e. column, in first for loop. Store in dict.
#Then sum weights for each (bin,cat) pair. Store in dict as,
sumwProcMap = {}
sumwProcCatMapReco = {}
sumwProcCatMapPred = {}
for iProc in range(nClasses+1):
    sumwProcMap[iProc] = np.sum(diphoFW*(diphoJ==iProc))
    for jProc in range(nClasses):
        sumwProcCatMapReco[(iProc,jProc)] = np.sum(diphoFW*(diphoJ==iProc)*(diphoR==jProc))


#Sum weights for entire predicted catgeory i.e. row for BDT pred cat and Reco cat
sumwCatMapReco = {}
sumwCatMapPred = {}
for iProc in range(nClasses): 
    sumwCatMapReco[iProc] = np.sum(diphoFW*(diphoR==iProc))

#Set 2D hist axis/bin labels
for iBin in range(nClasses+1):
    procHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistReco.GetXaxis().SetTitle('STXS ggH process')

    procHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistPred.GetXaxis().SetTitle('STXS ggH process')

    catHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )                                               
    catHistReco.GetXaxis().SetTitle('STXS ggH process')

    catHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistPred.GetXaxis().SetTitle('STXS ggH process')

for iBin in range(nClasses):
    procHistReco.GetYaxis().SetBinLabel( iBin+1, catNames[iBin] )
    procHistReco.GetYaxis().SetTitle('Event category')

    procHistPred.GetYaxis().SetBinLabel( iBin+1, catNames[iBin] )
    procHistPred.GetYaxis().SetTitle('Event category')

    catHistReco.GetYaxis().SetBinLabel( iBin+1, catNames[iBin] )
    catHistReco.GetYaxis().SetTitle('Event category')

    catHistPred.GetYaxis().SetBinLabel( iBin+1, catNames[iBin] )
    catHistPred.GetYaxis().SetTitle('Event category')

#Set up color pallete and other canvas options
npoints = 2
stops = [0.00,1.00]
red = [1.00, 0.00]
green = [1.00, 0.60]
blue = [1.00, 0.70]
ncontours = 256
alpha=1.

stops = array('d',stops)
red = array('d',red)
green = array('d',green)
blue = array('d',blue)

r.TColor.CreateGradientColorTable(npoints, stops, red, green, blue, ncontours, alpha)
r.gStyle.SetNumberContours(256)
#r.gStyle.SetGridColor(16)

#Fill 2D hists with percentage of events
for iProc in range(nClasses+1):
    for jProc in range(nClasses):
        #Indiv bin entries for reco and pred, normalised by sum of bin i.e. sum of col
        procWeightReco = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwProcMap[iProc]
        if procWeightReco < 0.5: procWeightReco=-1 
        #Indiv bin entries for reco and pred normalised to sum of cat i.e. sum of row
        catWeightReco  = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwCatMapReco[jProc]
        if catWeightReco < 0.5: catWeightReco=-1 

        procHistReco.Fill(iProc, jProc, procWeightReco)
        catHistReco.Fill(iProc, jProc, catWeightReco)

#draw and save the 2D hists
canv = r.TCanvas()
canv.SetRightMargin(0.15)
canv.SetLeftMargin(0.12)
canv.SetBottomMargin(0.18)
canv.SetRightMargin(0.15)
r.gStyle.SetPaintTextFormat('2.1f')
catHistReco.Draw('colz,text')
v_line = r.TLine(8.5,-0.5,8.5,8.5)
v_line.SetLineColorAlpha(r.kGray,0.8)
v_line.Draw()
v_line_dark = r.TLine(9.5,-0.5,9.5,8.5)
v_line_dark.SetLineColor(r.kBlack)
v_line_dark.Draw()
h_line_dark = r.TLine(-0.5,8.5,9.5,8.5)
h_line_dark.SetLineColor(r.kBlack)
h_line_dark.Draw()
catHistReco.GetXaxis().SetTitleOffset(2.4)
catHistReco.GetXaxis().SetTickLength(0)
catHistReco.GetXaxis().LabelsOption('v')
catHistReco.GetYaxis().SetTitleOffset(1.6)
catHistReco.GetZaxis().SetTitleOffset(1.2)
catHistReco.SetMaximum(100)
catHistReco.SetMinimum(0)
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='35.9 fb^{-1} (2016)')
canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/purityMatricesForRobustness/dataSigs_reco.png')

'''
#NOTE: will need to change this for each weight scenario
if opts.className:
  if 'Jet' in opts.className:
    canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/BDT/NoW/nJetBDTPurityMatrix_NoW.pdf')
  elif 'nClass' in opts.className:
    canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nClassOutputs/BDT/NoW/nClassBDTPurityMatrix_NoW.pdf')
else: #reco
  canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/PurityMatrixReco.pdf')
'''

canv = r.TCanvas()
canv.SetRightMargin(0.15)
canv.SetLeftMargin(0.12)
canv.SetBottomMargin(0.18)
canv.SetRightMargin(0.15)
#canv.SetGridx()
r.gStyle.SetPaintTextFormat('2.1f')
prettyHist(procHistReco)
procHistReco.Draw('colz,text')
v_line = r.TLine(8.5,-0.5,8.5,8.5)
v_line.SetLineColorAlpha(r.kGray,0.8)
v_line.Draw()
v_line_dark = r.TLine(9.5,-0.5,9.5,8.5)
v_line_dark.SetLineColor(r.kBlack)
v_line_dark.Draw()
h_line_dark = r.TLine(-0.5,8.5,9.5,8.5)
h_line_dark.SetLineColor(r.kBlack)
h_line_dark.Draw()
procHistReco.GetXaxis().SetTitleOffset(2.4)
procHistReco.GetXaxis().SetTickLength(0)
procHistReco.GetXaxis().LabelsOption('v')
procHistReco.GetYaxis().SetTitleOffset(1.6)
procHistReco.GetZaxis().SetTitleOffset(1.2)
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='35.9 fb^{-1} (2016)')

'''
#NOTE: will need to change this for each weight scenario
if opts.className:
  if 'Jet' in opts.className:
    canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/BDT/NoW/nJetBDTPurityMatrixNormByProc_NoW.pdf')
  elif 'nClass' in opts.className:
    canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nClassOutputs/BDT/NoW/nClassBDTPurityMatrixNormByProc_NoW.pdf')
else: #reco
  canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/PurityMatrixRecoNormByProc.pdf')
'''


                              ##### plot category efficiencies as radar plots #####
recoClass  = trainTotal['reco'].values
weights    = trainTotal['weight'].values

    #BDT:
correctDict   = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
incorrectDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}

for true, guess, weight in zip(diphoJ, diphoR, weights): #was trueClass,recoClass
  if true==guess:
    correctDict[true].append(weight)
  else:
    incorrectDict[true].append(weight)

correctList   = []
incorrectList = []

#sum the weights in the dict for each cat
for iCat in range(len(correctDict.keys())):
  correctList.append(sum(correctDict[iCat]))

for iCat in range(len(incorrectDict.keys())-4):
  incorrectList.append(sum(incorrectDict[iCat]))

#convert to numpy for pyplot
correctArray   = np.asarray(correctList)
incorrectArray = np.asarray(incorrectList)

print('correctArray')
print(correctArray)

print('incorrectArray')
print(incorrectArray)

effArrayMVA = correctArray/(correctArray+incorrectArray)
print('MVA Effs are:')
print(effArrayMVA)
print('\nAverage MVA eff is: %f'%effArrayMVA.mean())

    #reco:
#Create and fill correct and incorrect dicts
correctDict   = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
incorrectDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}

for true, guess, weight in zip(diphoJ, recoClass, weights): #was trueClass,recoClass
  if true==guess:
    correctDict[true].append(weight)
  else:
    incorrectDict[true].append(weight)

correctList   = []
incorrectList = []



#sum the weights in the dict for each cat
for iCat in range(len(correctDict.keys())):
  correctList.append(sum(correctDict[iCat]))

for iCat in range(len(incorrectDict.keys())-4):
  incorrectList.append(sum(incorrectDict[iCat]))

#convert to numpy for pyplot
correctArray   = np.asarray(correctList)
incorrectArray = np.asarray(incorrectList)

print('correctArray')
print(correctArray)

print('incorrectArray')
print(incorrectArray)

effArrayReco = correctArray/(correctArray+incorrectArray)
print('reco Effs are:')
print(effArrayReco)
print('\nAverage Reco eff is: %f'%effArrayReco.mean())


#do the plotting
df = pd.DataFrame({
    'group':['BDT','Reco'],
    'Bin 1': [effArrayMVA[0], effArrayReco[0]],
    'Bin 2': [effArrayMVA[1], effArrayReco[1]],
    'Bin 3': [effArrayMVA[2], effArrayReco[2]],
    'Bin 4': [effArrayMVA[3], effArrayReco[3]],
    'Bin 5': [effArrayMVA[4], effArrayReco[4]],
    'Bin 6': [effArrayMVA[5], effArrayReco[5]],
    'Bin 7': [effArrayMVA[6], effArrayReco[6]],
    'Bin 8': [effArrayMVA[7], effArrayReco[7]],
    'Bin 9': [effArrayMVA[8], effArrayReco[8]] 
})

print(df)

# number of variables
#categories=list(df)[1:]
categories = ['0J low', '0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high', 'BSM']
N = len(categories)


# Calculate angle of axis (plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]


# Initialise the spider plot
ax = plt.subplot(111, polar=True)

#Add text box for axis label
plt.text(6, 0.35, 'Efficiency', color="grey")

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=12)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0,0.2,0.4,0.6,0.8,1], ["0.0", "0.2","0.4","0.6","0.8",""], color="grey", size=10)
plt.ylim(0,1)
 
  
# Plot data
#BDT
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="BDT", color='g')
ax.fill(angles, values, 'b', alpha=0.1)
 
# Reco
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Reco", color='r')
ax.fill(angles, values, 'r', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.14, 0.04))
plt.show()
#plt.savefig('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/reco/effRadarPlot.pdf')   

