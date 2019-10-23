#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import os
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
#parser.add_option('--equalWeights', default=False, action='store_true', help='Alter weights for training so that signal and background have equal sum of weights')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
#validFrac = 0.1 #NOTE: we don't do the validation, so test on 30% of data 
np.random.seed(86421)
weightScale= False

#get trees from files, put them in data frames
#NOTE: ttH file not available for 2017 so just train without this signal. Also dc about VH as small x-sec
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root'}
#procFileMap = {'ggh':'powheg_ggH.root', 'vbf':'powheg_VBF.root', 'tth':'powheg_ttH.root', 
               #'dipho':'Dipho.root', 'gjet_promptfake':'GJet_pf.root', 'gjet_fakefake':'GJet_ff.root', 'qcd_promptfake':'QCD_pf.root', 'qcd_fakefake':'QCD_ff.root'}
theProcs = procFileMap.keys()

#define the different sets of variables used
diphoVars  = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      print('reading tree for proc: %s' % proc)
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc)
      elif 'dipho' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_anyfake_13TeV_GeneralDipho'%proc)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dijet_*',0)
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
      del newTree
      del newFile
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  trainTotal = trainTotal[trainTotal.HTXSstage1_1cat!=100]
  print 'done basic preselection cuts'

  if weightScale:
    print('MC weights before were:')
    print(trainTotal['weight'].head(10))
    trainTotal.loc[:,'weight'] *= 1000
    print('weights after scaling are:') 
    print(trainTotal['weight'].head(10))
  
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
  trainTotal['diphoWeight'] = trainTotal.apply(diphoWeight,axis=1)
  trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/trainTotal.pkl'%frameDir)
  print 'frame saved as %s/trainTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe: %s/%s' % (frameDir,opts.dataFrame)

sigSumW = np.sum( trainTotal[trainTotal.HTXSstage1cat>0.01]['weight'].values )
bkgSumW = np.sum( trainTotal[trainTotal.HTXSstage1cat==0]['weight'].values )
print 'sigSumW %.6f'%sigSumW
print 'bkgSumW %.6f'%bkgSumW
print 'ratio %.6f'%(sigSumW/bkgSumW)
#exit('first just count the weights')

#define the indices shuffle (useful to keep this separate so it can be re-used)
theShape = trainTotal.shape[0]
diphoShuffle = np.random.permutation(theShape)
diphoTrainLimit = int(theShape*trainFrac)
#diphoValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for diphoton training (as numpy arrays)
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
diphoTW = trainTotal['diphoWeight'].values #weights for training sample
diphoAW = trainTotal['altDiphoWeight'].values #alternative weights for training sample
diphoFW = trainTotal['weight'].values #weights for test sample
diphoM  = trainTotal['dipho_mass'].values

del trainTotal

diphoX  = diphoX[diphoShuffle]
diphoY  = diphoY[diphoShuffle]
diphoTW = diphoTW[diphoShuffle]
diphoAW = diphoAW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoM  = diphoM[diphoShuffle]

diphoTrainX, diphoTestX  = np.split( diphoX,  [diphoTrainLimit] )
diphoTrainY, diphoTestY  = np.split( diphoY,  [diphoTrainLimit] )
diphoTrainTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit] )
diphoTrainAW, diphoTestAW = np.split( diphoAW, [diphoTrainLimit] )
diphoTrainFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit] )
diphoTrainM, diphoTestM  = np.split( diphoM,  [diphoTrainLimit] )

#build the background discrimination BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
trainParams = {}
trainParams['objective'] = 'binary:logistic'
trainParams['nthread'] = 1
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for params in opts.trainParams:
    pairs  = params.split(',')
    for pairs in pairs:
      key = pairs.split(':')[0]
      data = pairs.split(':')[1]
      trainParams[key] = data
      paramExt += '%s_%s__'%(key,data)
    paramExt = paramExt[:-2] 

#cross validate and chose models that gives smallest out of fold prediction error
#print 'starting cross validation'
#cvResult = xg.cv(trainParams, trainingDipho, num_boost_round=2000,  nfold = 4, early_stopping_rounds = 20, stratified = True, verbose_eval=True)                                                                   
#print('Best number of trees = {}'.format(cvResult.shape[0]))
#trainParams['n_estimators'] = cvResult.shape[0]

print 'about to train diphoton BDT with hyperparameters:%s'%(trainParams)
diphoModel = xg.train(trainParams, trainingDipho)
print 'done'

#save it if not doing HP optimisation
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)

diphoModel.save_model('%s/diphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/diphoModel%s.model'%(modelDir,paramExt)


#build same thing but with equalised weights
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)
#cross validate again for other alt model
#print 'starting cross validation'
#cvResult = xg.cv(trainParams, altTrainingDipho, num_boost_round=2000,  nfold = 4, early_stopping_rounds = 20, stratified = True, verbose_eval=True)                                                                   
#print('Best number of trees = {}'.format(cvResult.shape[0]))
#trainParams['n_estimators'] = cvResult.shape[0]



print 'about to train alternative diphoton BDT with hyperparameters:%s'%(trainParams)
altDiphoModel = xg.train(trainParams, altTrainingDipho)
print 'done'

#save it if not doing HP optimisation

altDiphoModel.save_model('%s/altDiphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/altDiphoModel%s.model'%(modelDir,paramExt)


#check performance of each training
diphoPredYxcheck = diphoModel.predict(trainingDipho)
diphoPredY = diphoModel.predict(testingDipho)
normROCtrain = roc_auc_score(diphoTrainY, diphoPredYxcheck, sample_weight=diphoTrainFW)
normROCtest = roc_auc_score(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
print 'Default training performance:'
print 'area under roc curve for training set = %1.3f'%(normROCtrain)
print 'area under roc curve for test set     = %1.3f'%(normROCtest)

altDiphoPredYxcheck = altDiphoModel.predict(trainingDipho)
altDiphoPredY = altDiphoModel.predict(testingDipho)
altROCtrain = roc_auc_score(diphoTrainY, altDiphoPredYxcheck, sample_weight=diphoTrainFW)
altROCtest = roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
print 'Alternative training performance:'
print 'area under roc curve for training set = %1.3f'%(altROCtrain)
print 'area under roc curve for test set     = %1.3f'%(altROCtest)


#save roc_auc and associated HPs for test and train sets iff better than other ones in text file
#check if the dir and file exists.  
if not path.isdir('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs'): 
 system('mkdir -p /vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs')
if not path.isfile('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/normROC_and_HP.txt'):
 system('touch /vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs/normROC_and_HP.txt')
with open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs/normROC_and_HP.txt','a+') as bestHPfile: 
  lines = bestHPfile.readlines()
  rocScores = []
  if len(lines) == 0: rocScores.append(0.000) 
  else:
    for line in lines:
      rocScores.append(float(line[:5]))
  print 'Norm roc list before appending:'
  print rocScores
  if(roc_auc_score(diphoTestY, diphoPredY, sample_weight=diphoTestFW) > rocScores[-1]) and (abs(normROCtrain - normROCtest)<0.01):
    print 'Difference between normal train and test scores: %1.3f not too large' % (abs(normROCtrain - normROCtest))
    print 'best value so far. Appending to file'
    bestHPfile.write('%1.3f  (ROC auc) HPs: %s \n' % (normROCtest,trainParams))

#same thing for alt model
if not path.isfile('%s/altROC_and_HP.txt' % os.getcwd()):
 system('touch /vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs/altROC_and_HP.txt')
with open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/diphoROCOutputs/altROC_and_HP.txt', 'a+') as bestHPfile: 
  lines = bestHPfile.readlines()
  rocScores = []
  if len(lines) == 0: rocScores.append(0.000) 
  else:
    for line in lines:
      rocScores.append(float(line[:5]))
  print 'Alt roc list before appending:'
  print rocScores
  if(roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW) > rocScores[-1]) and (abs(altROCtrain - altROCtest)<0.01):
    print 'Difference between alt train and test scores not too large: %1.3f' % (abs(altROCtrain - altROCtest))
    print 'best value so far. Appending to file'
    bestHPfile.write('%1.3f  (ROC auc). HPs: %s \n' % (altROCtest,trainParams))

'''
#same thing for alt model
altRocFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/rocOutputs/altRocs.txt','a+')  
altBestHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/rocOutputs/bestAltHPs.txt','a+')  
lines = altRocFile.readlines()
altRocScores = []
for line in lines:
  altRocScores.append(float(line))
print 'Alt roc scores before appending'
print altRocScores
if(roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW) > altRocScores[-1]):
  altRocFile.write('%1.3f\n' % roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW))
  print'best value so far. Appending to ALT file'
  altBestHPFile.write('HPs: %s --> roc_auc: %1.3f \n' % (trainParams,roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)))
altRocFile.close()
altBestHPFile.close()
'''

#exit("Plotting not working for now so exit")
#make some plots 
plotDir = trainDir.replace('trees','plots')
plotDir = '%s/%s'% (plotDir,paramExt)
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)
bkgEff, sigEff, nada = roc_curve(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
plt.figure(1)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('%s/diphoROC.pdf'%plotDir)
bkgEff, sigEff, nada = roc_curve(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
plt.figure(2)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('%s/altDiphoROC.pdf'%plotDir)
plt.figure(3)
xg.plot_importance(diphoModel)
#plt.show()
plt.savefig('%s/diphoImportances.pdf'%plotDir)
plt.figure(4)
xg.plot_importance(altDiphoModel)
#plt.show()
plt.savefig('%s/altDiphoImportances.pdf'%plotDir)

#draw sig vs background distribution (MC weights BDT)
nOutputBins = 50
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, diphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, diphoPredY, weights=bkgScoreW)

sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/sigScore.pdf'%plotDir)
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/bkgScore.pdf'%plotDir)
theCanv.Close()

#apply transformation to flatten ggH
for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    bkgVal = bkgScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
    else:
        bkgScoreHist.SetBinContent(iBin, 0)
        
sigScoreHist.Scale(1./sigScoreHist.Integral())
bkgScoreHist.Scale(1./bkgScoreHist.Integral())
sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist,same')
useSty.drawCMS()
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/outputScores.pdf'%plotDir)

#draw sig vs background distribution (alt BDT)
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, altDiphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, altDiphoPredY, weights=bkgScoreW)

sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/altSigScore.pdf'%plotDir)
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/altBkgScore.pdf'%plotDir)

for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    bkgVal = bkgScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
    else:
        bkgScoreHist.SetBinContent(iBin, 0)
        
sigScoreHist.Scale(1./sigScoreHist.Integral())
bkgScoreHist.Scale(1./bkgScoreHist.Integral())
sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist,same')
useSty.drawCMS()
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/altOutputScores.pdf'%plotDir)
