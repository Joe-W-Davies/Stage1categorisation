#usual imports
import ROOT as r
r.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass, jetPtToClass, jetWeight, jetPtToggHClass, sqrtJetWeight, cbrtJetWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from sklearn.metrics import accuracy_score, log_loss
from math import pi, log

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

#setup global variables. Note no validation fraction; validation done in CV
trainDir = checkDir(opts.trainDir)
frameDir = checkDir(trainDir.replace('trees','frames'))
modelDir = checkDir(trainDir.replace('trees','models'))
plotDir  = checkDir(trainDir.replace('trees','plots') + '/nJetCategorisation')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
nJetClasses = 3
nClasses = 9 
binNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low low', '2J low med', '2J low high', 'BSM']
weightScale = False #scale weights to be closer to one

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

classVars  = ['n_rec_jets','dijet_Mjj','diphopt',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

jetVars  = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

#get trees from files, put them in data frames
procFileMap = {'ggh':'Merged.root'}
theProcs = procFileMap.keys()

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc:
        trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
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
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.stage1cat>-1.]
  trainTotal = trainTotal[trainTotal.dijet_Mjj<350]
  print 'done basic preselection cuts'
 
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['truthClass'] = trainTotal.apply(truthClass, axis=1)
  trainTotal['truthJets'] = trainTotal.apply(truthJets, axis=1)
  trainTotal['jetWeight'] = trainTotal.apply(jetWeight, axis=1)
  trainTotal['sqrtJetWeight'] = trainTotal.apply(sqrtJetWeight, axis=1)
  trainTotal['cbrtJetWeight'] = trainTotal.apply(cbrtJetWeight, axis=1)
  print 'all columns added'

  #only select processes relevant for nJet training
  trainTotal = trainTotal[trainTotal.truthJets>-1]
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal = trainTotal[trainTotal.truthClass!=-1]
  #Remove vbf-like classes as don't care about predicting them
  # onlt do this at gen level in training!
  trainTotal = trainTotal[trainTotal.truthClass<9]
 
  if weightScale:
    print('MC weights before were:')
    print(trainTotal['weight'].head(10))
    trainTotal.loc[:,'weight'] *= 1000
    print('weights after scaling are:') 
    print(trainTotal['weight'].head(10))
  
  #save as a pickle file
  trainTotal.to_pickle('%s/jetTotal.pkl'%frameDir)
  print 'frame saved as %s/jetTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle(opts.dataFrame)
  print 'Successfully loaded the dataframe'

sumW_0J = np.sum( trainTotal[trainTotal.truthJets==0]['weight'].values )
sumW_1J = np.sum( trainTotal[trainTotal.truthJets==1]['weight'].values )
sumW_2J = np.sum( trainTotal[trainTotal.truthJets==2]['weight'].values )
print '0J sum of weights is %.6f, frac is %.6f'%(sumW_0J, sumW_0J/(sumW_0J+sumW_1J+sumW_2J))
print '1J sum of weights is %.6f, frac is %.6f'%(sumW_1J, sumW_1J/(sumW_0J+sumW_1J+sumW_2J))
print '2J sum of weights is %.6f, frac is %.6f'%(sumW_2J, sumW_2J/(sumW_0J+sumW_1J+sumW_2J))

#check sum of weights for the classes being predicted are equal
sumJW_0J = np.sum( trainTotal[trainTotal.truthJets==0]['jetWeight'].values ) 
print 'Sum of weights for class %i: %.5f' %(0,sumJW_0J) 
sumJW_1J = np.sum( trainTotal[trainTotal.truthJets==1]['jetWeight'].values ) 
print 'Sum of weights for class %i: %.5f' %(1,sumJW_1J) 
sumJW_2J = np.sum( trainTotal[trainTotal.truthJets==2]['jetWeight'].values ) 
print 'Sum of weights for class %i: %.5f' %(2,sumJW_2J) 

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)

#setup the various datasets for multiclass training. 
classI        = trainTotal[jetVars].values
classJ        = trainTotal['truthJets'].values
classMjjTruth = trainTotal['gen_dijet_Mjj'].values
classJW       = trainTotal['sqrtJetWeight'].values #Depends on the weight scenario
classFW       = trainTotal['weight'].values
classM        = trainTotal['CMS_hgg_mass'].values
classN        = trainTotal['n_rec_jets'].values
classP        = trainTotal['diphopt'].values
classR        = trainTotal['reco'].values
classY        = trainTotal['truthClass'].values

#shuffle datasets
classI        = classI[classShuffle]
classJ        = classJ[classShuffle]
classMjjTruth = classMjjTruth[classShuffle]
classJW       = classJW[classShuffle]
classFW       = classFW[classShuffle]
classM        = classM[classShuffle]
classN        = classN[classShuffle]
classP        = classP[classShuffle]
classR        = classR[classShuffle]
classY        = classY[classShuffle]

# Include info for 1.1 categorisation for debugging
classGenPtH   = trainTotal['gen_pTH'].values
classMjj      = trainTotal['dijet_Mjj'].values
classGenMjj   = trainTotal['gen_dijet_Mjj'].values
classPtHjj    = trainTotal['ptHjj'].values
classGenPtHjj = trainTotal['gen_ptHjj'].values

#Then shuffle them
classGenPtH   = classGenPtH[classShuffle]
classMjj      = classMjj[classShuffle]       
classGenMjj   = classGenMjj[classShuffle]    
classPtHjj    = classPtHjj[classShuffle]     
classGenPtHjj = classGenPtHjj[classShuffle]  


#split datasets
classTrainI, classTestI  = np.split( classI,  [classTrainLimit] )
classTrainJ, classTestJ  = np.split( classJ,  [classTrainLimit] )
classTrainMjjTruth, classTestMjjTruth = np.split( classMjjTruth, [classTrainLimit] )
classTrainJW, classTestJW  = np.split( classJW,  [classTrainLimit] )
classTrainFW, classTestFW  = np.split( classFW,  [classTrainLimit] )
classTrainM,  classTestM  = np.split( classM,  [classTrainLimit] )
classTrainN,  classTestN  = np.split( classN,  [classTrainLimit] )
classTrainP,  classTestP  = np.split( classP,  [classTrainLimit] )
classTrainR,  classTestR  = np.split( classR,  [classTrainLimit] )
classTrainY,  classTestY  = np.split( classY,  [classTrainLimit] )

#Do the same but with the info used later in the cross check
classTrainGenPtH, classTestGenPtH  = np.split( classGenPtH,  [classTrainLimit] )
classTrainMjj, classTestMjj  = np.split( classMjj,  [classTrainLimit] )
classTrainGenMjj,  classTestGenMjj  = np.split( classGenMjj,  [classTrainLimit] )
classTrainPtHjj,  classTestPtHjj  = np.split( classPtHjj,  [classTrainLimit] )
classTrainGenPtHjj,  classTestGenPtHjj  = np.split( classGenPtHjj,  [classTrainLimit] )

#build the jet-classifier
trainingJet = xg.DMatrix(classTrainI, label=classTrainJ, weight=classTrainJW, feature_names=jetVars)
testingJet  = xg.DMatrix(classTestI,  label=classTestJ,  weight=classTestFW, feature_names=jetVars)
trainParams = {}
trainParams['objective'] = 'multi:softprob'
trainParams['num_class'] = nJetClasses
trainParams['nthread'] = 1
trainParams['eval_metric'] = 'mlogloss'

paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]

'''
print 'starting cross validation'
cvResult = xg.cv(trainParams, trainingJet, num_boost_round=2000,  nfold = 4, early_stopping_rounds = 50, stratified = True, verbose_eval=True , seed = 12345)
print('Best number of trees = {}'.format(cvResult.shape[0]))
trainParams['n_estimators'] = cvResult.shape[0]
'''

#Fit the model with best n_trees/estimators
print('Fitting on the training data')
jetModel = xg.train(trainParams, trainingJet) 
print('Done')

#save
'''
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
jetModel.save_model('%s/nJetModelWithSqrtEQWeights%s.model'%(modelDir,paramExt))
print 'saved as %s/jetModelWithSqrtEQWeightsWeights%s.model'%(modelDir,paramExt)
'''

#get predicted values. Can include priors here
predProbJet = jetModel.predict(testingJet).reshape(classTestJ.shape[0],nJetClasses) 
#totSumW = sumW_0J + sumW_1J +sumW_2J 
#priors = np.array( [sumW_0J/totSumW, sumW_1J/totSumW, sumW_2J/totSumW] ) 
#predProbJet *= priors 
classPredJ = np.argmax(predProbJet, axis=1) 

#create dataframe to do pred cats
predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(classTestP), pd.DataFrame(classTestPtHjj), pd.DataFrame(classTestMjj) ], axis=1)
predFrame.columns = ['n_pred_jets', 'diphopt', 'ptHjj', 'dijet_Mjj']
predFrame['predClass'] = predFrame.apply(jetPtToggHClass, axis=1)
classPredY = predFrame['predClass'].values
print 'pred frame'
print(predFrame.head(10))

#cross check: save data frame to check everything is being printed binned and catgeorised correctly
testTotal = pd.concat([pd.DataFrame(classTestJ), pd.DataFrame(classTestGenPtH), pd.DataFrame(classTestGenPtHjj),pd.DataFrame(classTestGenMjj),pd.DataFrame(classTestY), pd.DataFrame(classTestN), predFrame, pd.DataFrame(classTestR) ], axis=1)
#rename cols so they match the defs in the row function
testTotal.columns = ['n_truth_jets','gen_pTH','gen_pTHjj','gen_Mjj', 'truthClass','n_rec_jets','n_pred_jets', 'diphopt', 'ptHjj', 'Mjj', 'pred_class','reco_class']
testTotal.to_pickle('%s/jetTotalModified.pkl'%frameDir)
print 'frame saved as %s/jetTotalModified.pkl'%frameDir


print
print 'reconstructed  number of jets =  %s'%classTestN.astype('int')
print 'BDT predicted  number of jets =  %s'%classPredJ
print 'truth          number of jets =  %s'%classTestJ
print 'BDT predicted  number of jets =  %s'%classPredJ
print 'reconstructed     diphoton pt =  %s'%classTestP

print '                   reco class =  %s' %classTestR
print '          BDT predicted class =  %s'%classPredY
print '                  truth class =  %s'%classTestY
print


#NOTE: Below, in the calcualtion of accuracy, efficiency, and the purity matrices,
# We haven't included the contribution from ggH vbf-like procs into the 9 reco categories
# Hence this is imcomplete. However, the contamination is small, so it can still serve as
# an approximation. Best to just calculate these values in the sigificances script, since there
# we don't remive the ggH VBF-like event at gen level, which we do do here

BDTaccuracy = accuracy_score(classTestY,classPredY)#,sample_weight=classTestFW)
print
print('Accuracy score for the BDT is: %.4f' %(BDTaccuracy)) 
print('Accuracy score for Reco is  : %.4f' %(accuracy_score(classTestY,classTestR,sample_weight=classTestFW)))
print

#Calculate the loss on the test set, rather than the accruacy and eff
#Need to evaluate the loss of predicting jet number rather than class since this was learning task
mLogLoss = log_loss(classTestJ,predProbJet,sample_weight=classTestJW)
print
print('BDT multiclass log loss is: %.4f' %(mLogLoss)) 
print

### Plotting ###

#define h_truth_class, h_pred_class FROM RECO, and h_right and h_wrong for eff hist
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nClasses,-0.5,nClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

#Label the x bins
for iBin in range(1, nClasses+1):
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
#effHist = r.TH1F

#Draw the historgrams
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/recoPredJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/recoPredJetHist%s.png'%(plotDir,paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/recoEfficiencyJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/recoEfficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/recoEfficiencyJetHist%s.pdf'%(paramExt))

'''
#print effeciency averages to text file for comparisons (reco) (first 9 bins only)
recoAveragesFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/reco_eff_averages.txt','a+')
recoAveragesHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/reco_eff_averages_HP.txt','a+')
eff_array = np.zeros(nClasses)
for iBin in range(1, nClasses+1):
  #print 'Bin %g: %f'%(iBin, rightHist.GetBinContent(iBin))
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
lines = recoAveragesFile.readlines()
recoAverages = []
for line in lines:
  recoAverages.append(float(line))
if (np.average(eff_array) > recoAverages[-1]):
  recoAveragesFile.write('%1.3f\n' % np.average(eff_array)) 
  recoAveragesHPFile.write('HPs: %s --> avg eff: %1.3f \n' % (trainParams, np.average(eff_array)) ) 
recoAveragesFile.close()
'''

#setup more 1D hists for truth class, BDT PREDICTED class, right, and wrong
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nClasses,-0.5,nClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

#Label and fill bins
for iBin in range(1, nClasses+1):
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
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/predJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/predJetHist%s.png'%(plotDir,paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
canv.Print('%s/efficiencyJetHist%s.pdf'%(plotDir,paramExt))
canv.Print('%s/efficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/efficiencyJetHist%s.pdf'%(paramExt))

'''
#print some stat params to text file for comparisons (BDT pred) (first 9 bins only)
predAveragesFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/pred_eff_averages.txt','a+')
predAveragesHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetOutputs/pred_eff_averages_HP.txt','a+')
eff_array = np.zeros(nClasses)
for iBin in range(1, nClasses+1):
  #print 'Bin %g: %f'%(iBin, rightHist.GetBinContent(iBin))
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
lines = predAveragesFile.readlines()
predAverages = []
for line in lines:
  predAverages.append(float(line))
if (np.average(eff_array) > predAverages[-1]):
  predAveragesFile.write('%1.3f\n' % np.average(eff_array)) 
  predAveragesHPFile.write('HPs: %s --> avg eff: %1.3f \n' % (trainParams, np.average(eff_array)) ) 
predAveragesFile.close()
predAveragesHPFile.close()
'''


#print log-loss to text file, if smallest (in HP optimiastion). Print other params too for interest
#lossFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nJetBDT/losses_EW.txt','a+')
#lossHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nJetBDT/losses_EW.txt','a+')
#eff_array = np.zeros(nClasses)
#for iBin in range(1, nClasses+1):
#  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
#print('BDT avg eff: %1.4f' %(np.average(eff_array)))
#lines = lossFile.readlines()
#lossList = []
#for line in lines:
#  lossList.append(float(line))
#if (mLogLoss < lossList[-1]):
#  lossFile.write('%1.4f\n' % mLogLoss) 
#  lossHPFile.write('HPs: %s --> loss: %1.4f. acc:%1.4f. <eff>: %1.4f\n' % (trainParams,mLogLoss,BDTaccuracy,np.average(eff_array)) ) 
#lossFile.close()
#lossHPFile.close()

#declare 2D hists
nBinsX=nClasses
nBinsY=nClasses
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
for iProc in range(nClasses):
    sumwProcMap[iProc] = np.sum(classTestFW*(classTestY==iProc))
    for jProc in range(nClasses):
        sumwProcCatMapPred[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classPredY==jProc))
        sumwProcCatMapReco[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classTestR==jProc))


#Sum weights for entire predicted catgeory i.e. row for BDT pred cat and Reco cat
sumwCatMapReco = {}
sumwCatMapPred = {}
for iProc in range(nClasses):
    sumwCatMapPred[iProc] = np.sum(classTestFW*(classPredY==iProc))
    sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc))
    #sumwCatMapPred[iProc] = np.sum(classTestFW*(classTred==iProc)*(classTestY!=0)) #don't count bkg here
    #sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc)*(classTestY!=0))

#Set 2D hist axis/bin labels
for iBin in range(nClasses):
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
for iProc in range(nClasses):
    for jProc in range(nClasses):
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
canv.Print('%s/procJetHistReco%s.pdf'%(plotDir,paramExt))
canv.Print('%s/procJetHistReco%s.png'%(plotDir,paramExt))
prettyHist(catHistReco)
catHistReco.Draw('colz,text')
canv.Print('%s/catJetHistReco%s.pdf'%(plotDir,paramExt))
canv.Print('%s/catJetHistReco%s.png'%(plotDir,paramExt))
prettyHist(procHistPred)
procHistPred.Draw('colz,text')
canv.Print('%s/procJetHistPred%s.pdf'%(plotDir,paramExt))
canv.Print('%s/procJetHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/bestModelParams/nJetBDT/EQWeights/PurityMatrixNormByCol.pdf')
prettyHist(catHistPred)
catHistPred.Draw('colz,text')
canv.Print('%s/catJetHistPred%s.pdf'%(plotDir,paramExt))
canv.Print('%s/catJetHistPred%s.png'%(plotDir,paramExt))


# get feature importances
plt.figure(1)
xg.plot_importance(jetModel)
plt.show()
plt.savefig('%s/classImportances%s.pdf'%(plotDir,paramExt))
plt.savefig('%s/classImportances%s.png'%(plotDir,paramExt))
#plt.savefig('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/classImportances%s.pdf'%(paramExt))


### Radar plots ###
#Fill correct and incorrect dicts
correctDict   = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
incorrectDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

for true, guess, weight in zip(classTestY, classPredY, classTestFW):
  if true==guess:
    correctDict[true].append(weight)
  else:
    incorrectDict[true].append(weight)

#append to list then convert to np array  
correctList   = []
incorrectList = []

#sum the weights in the dict for each cat
for iCat in range(len(correctDict.keys())):
  correctList.append(sum(correctDict[iCat]))
  incorrectList.append(sum(incorrectDict[iCat]))

#convert to numpy for pyplot
correctArray   = np.asarray(correctList)
incorrectArray = np.asarray(incorrectList)

#calculate efficiency
effArrayBDT = correctArray/(correctArray+incorrectArray)

#NB this metric does not care about number of events in cats, unlike accuracy
print('BDT effs are:')
print(effArrayBDT)
print('\nAverage BDT eff is: %f'%effArrayBDT.mean())

#Do same thing for reco so we can compare
#Fill correct and incorrect dicts
correctDict   = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
incorrectDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

for true, guess, weight in zip(classTestY, classTestR, classTestFW):
  if true==guess:
    correctDict[true].append(weight)
  else:
    incorrectDict[true].append(weight)

correctList   = []
incorrectList = []

#sum the weights in the dict for each cat
for iCat in range(len(correctDict.keys())):
  correctList.append(sum(correctDict[iCat]))
  incorrectList.append(sum(incorrectDict[iCat]))

#convert to numpy for pyplot
correctArray   = np.asarray(correctList)
incorrectArray = np.asarray(incorrectList)

effArrayReco = correctArray/(correctArray+incorrectArray)
print('reco Effs are:')
print(effArrayReco)
print('\nAverage Reco eff is: %f'%effArrayReco.mean())

#do plotting
df = pd.DataFrame({
    'group':['BDT','Reco'],
    'Bin 1': [effArrayBDT[0], effArrayReco[0]],
    'Bin 2': [effArrayBDT[1], effArrayReco[1]],
    'Bin 3': [effArrayBDT[2], effArrayReco[2]],
    'Bin 4': [effArrayBDT[3], effArrayReco[3]],
    'Bin 5': [effArrayBDT[4], effArrayReco[4]],
    'Bin 6': [effArrayBDT[5], effArrayReco[5]],
    'Bin 7': [effArrayBDT[6], effArrayReco[6]],
    'Bin 8': [effArrayBDT[7], effArrayReco[7]],
    'Bin 9': [effArrayBDT[8], effArrayReco[8]] 
})

print(df)

# number of variables
categories=list(df)[1:]
N = len(categories)


# Calculate angle of axis (plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]


# Initialise the spider plot
plt.figure(2)
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=7)
plt.ylim(0,1)

# Plot data
#BDT
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="BDT", color='g')
ax.fill(angles, values, 'g', alpha=0.1)

 
# Reco
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Reco", color='r')
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
#plt.savefig('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/bestModelParams/nJetBDT/EQWeights/JetBDTRadarPlot.pdf')

