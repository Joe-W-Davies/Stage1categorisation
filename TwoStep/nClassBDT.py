#usual imports
import sys
import ROOT as r
r.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import xgboost as xg
from xgboost.callback import print_evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import uproot as upr
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn import cross_validation
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass1p2, jetPtToClass, procWeight, applyLumiScale
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from keras.utils import np_utils

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
(opts,args)=parser.parse_args()

def checkDir( dName ):
  if dName.endswith('/'): dName = dName[:-1]
  if not path.isdir(dName): 
    system('mkdir -p %s'%dName)
  return dName

#setup global variables
trainDir = checkDir(opts.trainDir)
frameDir = checkDir(trainDir.replace('trees','frames'))
modelDir = checkDir(trainDir.replace('trees','models'))
plotDir  = checkDir(trainDir.replace('trees','plots') + '/nClassCategorisation')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

trainToTestRatio = 0.7
sampleFrac = 1.0
nGGHClasses = 9
sampleFrame = False
weightScale = True

binNames = ['0J low','0J high','1J low','1J med','1J high','2J low low','2J low med','2J low high','BSM'] 

#define the different sets of variables used

allVars   = ['n_jet_30','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubleadJPt','dijet_SubsubleadJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'CosPhi','vtxprob','sigmarv','sigmawv' ,'dipho_mass','dipho_PToM', 'dipho_dijet_ptHjj',
              'HTXSstage1p2bin', 'weight', 'year_2016', 'year_2017', 'year_2018']

trainVars   = ['n_jet_30','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubleadJPt','dijet_SubsubleadJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'CosPhi','vtxprob','sigmarv','sigmawv','diphopt']


#procFileMap = {'ggh':'ggH_powheg_jetinfo.root'}
#procFileMap = {'ggh':'ggH_powheg_combined.root'}

procFileMap = {'ggh':'ggh_powheg_withBinaryYear_2016.root','ggh':'ggh_powheg_withBinaryYear_2017.root','ggh':'ggh_powheg_withBinaryYear_2018.root'}

theProcs = procFileMap.keys()
signals = ['ggh']
yearToLumi = {'2016':35.9, '2017':41.5, '2018':59.7}

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      print 'reading proc: %s' % proc
      if proc in signals:
          trainFile = upr.open('%s/%s'%(trainDir,fn)) 
          trainTree = trainFile['%s_125_13TeV_GeneralDipho'%proc]
          trainFrames[proc] = trainTree.pandas.df(allVars) 
      else: raise Exception('Error did not recognise process %s !'%proc)

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
  trainTotal = trainTotal[trainTotal.HTXSstage1p2bin!=-100]   
  trainTotal = trainTotal[trainTotal.dijet_Mjj<350]
  print 'done basic preselection cuts'
 
  #add extra info to dataframe  
  print 'about to add extra columns'
  trainTotal['truthClass'] = trainTotal.apply(truthClass1p2, axis=1, args=[True]) 
  #Remove vbf-like classes as don't care about predicting them
  # only do this at gen level in training!
  trainTotal = trainTotal[trainTotal.truthClass<9]
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['procWeight'] = trainTotal.apply(procWeight, axis=1)
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho, axis=1, args=[signals])
  print 'all columns added'

  #only select processes relevant for nJet training                                                             
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal = trainTotal[trainTotal.truthClass!=-1]

  #save procs as string (error when re-reading otherwise)
  trainTotal.loc[:, 'proc'] = trainTotal['proc'].astype(str)  
  #save as a pickle file
  #trainTotal.to_hdf('%s/MultiClassTotal.h5'%frameDir, key='df', mode='w')
  print 'frame saved as %s/MultiClassTotal.h5'%frameDir
  
  #scale by lumi based on year of data
  trainTotal['weight'] = trainTotal.apply(applyLumiScale, axis=1, args=[yearToLumi])
 
  if weightScale:
    print('MC weights before were:')
    print(trainTotal['weight'].head(10))
    trainTotal.loc[:,'weight'] *= 1000
    print('weights after scaling are:') 
    print(trainTotal['weight'].head(10))
   
 

#read in dataframe if above steps done once before
else:
  trainTotal = pd.read_hdf('%s'%opts.dataFrame, 'df')
  print 'Successfully loaded the dataframe \n'
  print(trainTotal.head(20))


#Sample dataframe randomly, if we want to test computation time
if sampleFrame:
  trainTotal = trainTotal.sample(frac=sampleFrac, random_state=1)

procWeightDict = {}
for iProc in range(nGGHClasses):  #from zero to 8 are ggH bins
  sumW = np.sum(trainTotal[trainTotal.truthClass==iProc]['weight'].values)
  sumW_proc = np.sum(trainTotal[trainTotal.truthClass==iProc]['procWeight'].values)
  procWeightDict[iProc] = sumW
  print 'Checling sum of proc (equal) weights for ggH STXS bin %i is: %.2f' %  (iProc,sumW_proc)  
  print 'Frac is %.6f' % (sumW/ (np.sum(trainTotal['weight'].values)))
  print 'Sum of proc weights for bin %i is: %.5f' % (iProc,sumW_proc)

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainToTestRatio)

#setup the various datasets for multiclass training
classI        = trainTotal[trainVars].values
classProcW    = trainTotal['procWeight'].values #changes depending on the weight scenario
classFW       = trainTotal['weight'].values
classR        = trainTotal['reco'].values
classY        = trainTotal['truthClass'].values

#shuffle datasets
classI        = classI[classShuffle]
classFW       = classFW[classShuffle]
classProcW    = classProcW[classShuffle]
classR        = classR[classShuffle]
classY        = classY[classShuffle]

#split datasets
classTrainI, classTestI     = np.split( classI,  [classTrainLimit] )
classTrainFW, classTestFW  = np.split( classFW, [classTrainLimit] )
classTrainProcW, classTestProcW  = np.split( classProcW, [classTrainLimit] )
classTrainR, classTestR  = np.split( classR,  [classTrainLimit] )
classTrainY, classTestY  = np.split( classY,  [classTrainLimit] )

#build the category classifier
trainingMC = xg.DMatrix(classTrainI, label=classTrainY, weight=classTrainFW, feature_names=trainVars)
testingMC  = xg.DMatrix(classTestI, label=classTestY, weight=classTestFW, feature_names=trainVars)

trainParams = {}
trainParams['objective'] = 'multi:softprob'
trainParams['num_class'] = nGGHClasses
trainParams['nthread'] = 1
trainParams['eval_metric'] = 'mlogloss' #loss function to be minimised

paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]
#else:
#  nClassesModel = xg.Booster()
#  nClassesModel.load_model('/vols/cms/jwd18/Stage1categorisation/Pass1/2016/models/nClassesModelMCWeights___min_child_weight_13__n_estimators_18__sub_sample_0.9030__eta_0.76__colsample_bytree_0.95__max_depth_12__gamma_1.85__lambda_1.8283.model')

#Time the training
startTime = time.time()

#cross validate and chose models that gives smallest out of fold prediction error
#print 'starting cross validation'
#cvResult = xg.cv(trainParams, trainingMC, num_boost_round=2000,  nfold = 4, early_stopping_rounds = 50, stratified = True, verbose_eval=True , seed = 12345)
#print('Best number of trees = {}'.format(cvResult.shape[0]))
#trainParams['n_estimators'] = cvResult.shape[0]


#Fit the model with best n_trees/estimators
print('Fitting on the training data')
nClassesModel = xg.train(trainParams, trainingMC) 
print('Done')


endTime = time.time()


#save. NOTE: don't do this when doing random HP search or will save thousands of models
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
nClassesModel.save_model('%s/nClassesModelMCW_Combined_%s.model'%(modelDir,paramExt))
print 'saved as %s/nClassesModelMCW_Combined_%s.model'%(modelDir,paramExt)


# Use the priors to tell the probs something about the abundance/likelihood of procs
'''
predProbTraining = nClassesModel.predict(trainingMC)
predProbClass = nClassesModel.predict(testingMC).reshape(classTestY.shape[0],nGGHClasses)
totSumW =  np.sum(trainTotal['weight'].values)
priors = [] #easier to append to list than numpy array. Then just convert after
for i in range(nGGHClasses): 
  priors.append(procWeightDict[i]/totSumW) 
predProbClass *= np.asarray(priors) #this is part where include class frac, not MC frac
predProbTraining *= np.asarray(priors) #this is part where include class frac, not MC frac
classPredY = np.argmax(predProbClass, axis=1) 
predClassTraining = np.argmax(predProbTraining, axis=1) 
'''

#No priors

predProbTraining = nClassesModel.predict(trainingMC)
predClassTraining = np.argmax(predProbTraining, axis=1) 

predProbClass = nClassesModel.predict(testingMC).reshape(classTestY.shape[0],nGGHClasses)
print(nClassesModel.predict(testingMC))

classPredY = np.argmax(predProbClass, axis=1) 


print
print '                   reco class =  %s' %classTestR
print '          BDT predicted class =  %s'%classPredY
print '                  truth class =  %s'%classTestY
print '          BDT accuracy train score =  %.4f' %accuracy_score(classTrainY,predClassTraining, sample_weight=classTrainFW)
print '          BDT accuracy test score =  %.4f' %accuracy_score(classTestY,classPredY, sample_weight=classTestFW)
print '         Reco accuracy test score =  %.4f' %accuracy_score(classTestY,classTestR, sample_weight=classTestFW) 


#Calculate the log loss
#Note that log loss requires class probabilities, not hard event class predictions
#For this reason, we can't compute log-loss on reco since it gives hard class predictions only
#FIXME: may have to redefinine testMC without the MC weights again here
LLClassProbY = nClassesModel.predict(testingMC)
mLogLoss = log_loss(classTestY,LLClassProbY,sample_weight=classTestFW)
BDTaccuracy = accuracy_score(classTestY,classPredY, sample_weight=classTestFW)

print 
print '          BDT log-loss=  %.4f' %mLogLoss
print



### Plotting ###

#define h_truth_class, h_pred_class FROM RECO, and h_right and h_wrong for eff hist
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nGGHClasses,-0.5,nGGHClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

#Label the x bins
for iBin in range(1, nGGHClasses+1):
    truthHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    predHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    rightHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    wrongHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])

#Fill truth class and RECO class and check if they are the same
for true,guess,w in zip(classTestY,classTestR,classTestFW):
    truthHist.Fill(true,w)
    predHist.Fill(guess,w)
    if true==guess: rightHist.Fill(true,w)
    else: wrongHist.Fill(true,w)
       
#ratio of bin i entries to bin 1 entries
firstBinVal = -1.
for iBin in range(1,truthHist.GetNbinsX()+1):
    if iBin==1: firstBinVal = truthHist.GetBinContent(iBin)
    ratio = float(truthHist.GetBinContent(iBin)) / firstBinVal
    print 'ratio of bin %g to bin 1 is %1.7f'%(iBin,ratio)
  #modify wrong and right hists to give efficiency i.e. right/right+wrong
wrongHist.Add(rightHist)
rightHist.Divide(wrongHist)

#Draw the historgrams
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/truthJetHist%s.pdf'%(paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/recoPredJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/recoPredJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/recoPredJetHist%s.pdf'%(paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS(onTop = True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/recoEfficiencyJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/recoEfficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/recoEfficiencyJetHist%s.pdf'%(paramExt))


#print some stat params to text file for comparisons (RECO) 
eff_array = np.zeros(nGGHClasses)
for iBin in range(1, nGGHClasses+1):
  #print 'Bin %g: %f'%(iBin, rightHist.GetBinContent(iBin))
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
print 'Reco average efficiency is: %1.3f\n' % (np.average(eff_array))


#setup more 1D hists for truth class, BDT PREDICTED class, right, and wrong
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nGGHClasses,-0.5,nGGHClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nGGHClasses,-0.5,nGGHClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

#Label and fill bins
for iBin in range(1, nGGHClasses+1):
    truthHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    predHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    rightHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    wrongHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])

for true,guess,w in zip(classTestY,classPredY,classTestFW):
    truthHist.Fill(true,w)
    predHist.Fill(guess,w)
    if true==guess: rightHist.Fill(true,w)
    else: wrongHist.Fill(true,w)
        
firstBinVal = -1.
for iBin in range(1,truthHist.GetNbinsX()+1):
    if iBin==1: firstBinVal = truthHist.GetBinContent(iBin)
    ratio = float(truthHist.GetBinContent(iBin)) / firstBinVal
    print 'ratio for bin %g is %1.7f'%(iBin,ratio)
   
#plot and save results for BDT PREDICTION 
wrongHist.Add(rightHist)
rightHist.Divide(wrongHist)
effHist = r.TH1F
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/truthJetHist%s.pdf'%(paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/predJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/predJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/truthJetHist%s.pdf'%(paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/efficiencyJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/efficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/efficiencyJetHist%s.pdf'%(paramExt))


#print log-loss to text file (used in HP optimiastion). Print other params too for interest
lossFile=open('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nClassBDT/losses_EQW.txt','a+')
lossHPFile=open('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nClassBDT/losses_EQW_HPs.txt','a+')
eff_array = np.zeros(nGGHClasses)
for iBin in range(1, nGGHClasses+1):
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
print('BDT avg eff: %1.4f' %(np.average(eff_array)))
lines = lossFile.readlines()
lossList = []
for line in lines:
  lossList.append(float(line))
if (mLogLoss < lossList[-1]):
  lossFile.write('%1.4f\n' % mLogLoss) 
  lossHPFile.write('HPs: %s --> loss: %1.4f. acc:%1.4f. <eff>: %1.4f\n' % (trainParams,mLogLoss,BDTaccuracy,np.average(eff_array)) ) 
lossFile.close()
lossHPFile.close()




#declare 2D hists
nBinsX=nGGHClasses
nBinsY=nGGHClasses
procHistReco = r.TH2F('procHistReco','procHistReco', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistReco.SetTitle('')
prettyHist(procHistReco)
procHistPred = r.TH2F('procHistPred','procHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistPred.SetTitle('')
prettyHist(procHistPred)
catHistReco  = r.TH2F('catHistReco','catHistReco', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistReco.SetTitle('')
prettyHist(catHistReco)
catHistPred  = r.TH2F('catHistPred','catHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistPred.SetTitle('')
prettyHist(catHistPred)

#generate weights for the 2D hists   

#Sum weights for each bin i.e. column, in first for loop. Store in dict.
#Then sum weights for each (bin,cat) pair. Store in dict as,
sumwProcMap = {}
sumwProcCatMapReco = {}
sumwProcCatMapPred = {}
for iProc in range(nGGHClasses):
    sumwProcMap[iProc] = np.sum(classTestFW*(classTestY==iProc))
    for jProc in range(nGGHClasses):
        sumwProcCatMapPred[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classPredY==jProc))
        sumwProcCatMapReco[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classTestR==jProc))


#Sum weights for entire predicted catgeory i.e. row for BDT pred cat and Reco cat
sumwCatMapReco = {}
sumwCatMapPred = {}
for iProc in range(nGGHClasses):
    sumwCatMapPred[iProc] = np.sum(classTestFW*(classPredY==iProc))
    sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc))
    #sumwCatMapPred[iProc] = np.sum(classTestFW*(classTred==iProc)*(classTestY!=0)) #don't count bkg here
    #sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc)*(classTestY!=0))

#Set 2D hist axis/bin labels
for iBin in range(nGGHClasses):
    procHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistReco.GetXaxis().SetTitle('gen bin')
    procHistReco.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistReco.GetYaxis().SetTitle('reco bin')

    procHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistPred.GetXaxis().SetTitle('gen bin')
    procHistPred.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistPred.GetYaxis().SetTitle('reco bin')

    catHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistReco.GetXaxis().SetTitle('gen bin')
    catHistReco.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistReco.GetYaxis().SetTitle('reco bin')

    catHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistPred.GetXaxis().SetTitle('gen bin')
    catHistPred.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistPred.GetYaxis().SetTitle('reco bin')

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
for iProc in range(nGGHClasses):
    for jProc in range(nGGHClasses):
        #Indiv bin entries for reco and pred, normalised by sum of bin i.e. sum of col
        procWeightReco = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwProcMap[iProc]
        procWeightPred = 100. * sumwProcCatMapPred[(iProc,jProc)] / sumwProcMap[iProc]
        
        #Indiv bin entries for reco and pred normalised to sum of cat i.e. sum of row
        catWeightReco  = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwCatMapReco[jProc]
        catWeightPred  = 100. * sumwProcCatMapPred[(iProc,jProc)] / sumwCatMapPred[jProc]

        procHistReco.Fill(iProc, jProc, procWeightReco)
        procHistPred.Fill(iProc, jProc, procWeightPred)
        catHistReco.Fill(iProc, jProc, catWeightReco)
        catHistPred.Fill(iProc, jProc, catWeightPred)

#draw and save the 2D hists
canv = r.TCanvas()
#canv.SetGrid()
r.gStyle.SetPaintTextFormat('2.0f')
prettyHist(procHistReco)
procHistReco.Draw('colz,text')
#canv.Print('%s/procJetHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procJetHistReco%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/procJetHistReco%s.pdf'%(paramExt))
prettyHist(catHistReco)
catHistReco.Draw('colz,text')
#canv.Print('%s/catJetHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catJetHistReco%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/purityMatricesForRobustness/recoFromTraining_%s.png'%(paramExt))
prettyHist(procHistPred)
procHistPred.Draw('colz,text')
#canv.Print('%s/procJetHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procJetHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/procJetHistPred%s.pdf'%(paramExt))
prettyHist(catHistPred)
catHistPred.Draw('colz,text')
#canv.Print('%s/catJetHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catJetHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/purityMatricesForRobustness/oldBDTBestHPs_%s.png'%(paramExt))

# get feature importances
plt.figure(1)
xg.plot_importance(nClassesModel)
plt.show()
plt.savefig('%s/classImportances%s.pdf'%(plotDir,paramExt))
plt.savefig('%s/classImportances%s.png'%(plotDir,paramExt))
#plt.savefig('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nClassBDT/classImportances.png')

print("--- %s seconds to cv ---" % (endTime - startTime))
