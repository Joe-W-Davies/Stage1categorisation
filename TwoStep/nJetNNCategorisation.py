#usual imports
import ROOT as r
r.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass, jetPtToClass, procWeight, jetPtToggHClass, sqrtJetWeight, cbrtJetWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.externals import joblib

#NN imports
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import *
from keras.optimizers import Nadam
from keras.optimizers import adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import h5py #import local version of this since CMSSW has old version

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
plotDir  = checkDir(trainDir.replace('trees','plots') + '/nJetCategorisation')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
validFrac = 0.1
nJetClasses = 3
nClasses = 9 
binNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low low', '2J low med', '2J low high', 'BSM']

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

#Set random seed for reproducibility
np.random.RandomState(12345)

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
  print 'done basic preselection cuts'

  #FIXME: trying to make datasig and df ehre more inline (since they should be the same)
  
 
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['truthClass'] = trainTotal.apply(truthClass, axis=1)
  trainTotal['truthJets'] = trainTotal.apply(truthJets, axis=1)
  trainTotal['jetWeight'] = trainTotal.apply(jetWeight, axis=1)
  #trainTotal['sqrtJetWeight'] = trainTotal.apply(sqrtJetWeight, axis=1)
  #trainTotal['cbrtJetWeight'] = trainTotal.apply(cbrtJetWeight, axis=1)
  print 'all columns added'

  #only select processes relevant for nJet training
  trainTotal = trainTotal[trainTotal.truthJets>-1]
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal = trainTotal[trainTotal.truthClass!=-1]
  #Remove vbf-like classes as don't care about predicting them (removes m_jj>350)
  trainTotal = trainTotal[trainTotal.truthClass<9]
  trainTotal = trainTotal.replace(-999,-10)  

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

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)
classValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for multiclass training
classI        = trainTotal[jetVars].values
classJ        = trainTotal['truthJets'].values
classMjjTruth = trainTotal['gen_dijet_Mjj'].values
classJW       = trainTotal['jetWeight'].values #changes depending on weight scenario
classFW       = trainTotal['weight'].values
classM        = trainTotal['CMS_hgg_mass'].values
classN        = trainTotal['n_rec_jets'].values
classP        = trainTotal['diphopt'].values
classR        = trainTotal['reco'].values
classY        = trainTotal['truthClass'].values

# Include gen level info for 1.1 categorisation for debugging
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

#split datasets
classTrainI, classValidI, classTestI     = np.split( classI,  [classTrainLimit,classValidLimit] )
classTrainJ, classValidJ, classTestJ     = np.split( classJ,  [classTrainLimit,classValidLimit] )
classTrainMjjTruth, classValidMjjTruth, classTestMjjTruth = np.split( classMjjTruth, [classTrainLimit,classValidLimit] )
classTrainJW, classValidJW, classTestJW  = np.split( classJW,  [classTrainLimit,classValidLimit] )
classTrainFW, classValidFW, classTestFW  = np.split( classFW,  [classTrainLimit,classValidLimit] )
classTrainM, classValidM, classTestM     = np.split( classM,  [classTrainLimit,classValidLimit] )
classTrainN, classValidN, classTestN     = np.split( classN,  [classTrainLimit,classValidLimit] )
classTrainP, classValidP, classTestP     = np.split( classP,  [classTrainLimit,classValidLimit] )
classTrainR, classValidR, classTestR     = np.split( classR,  [classTrainLimit,classValidLimit] )
classTrainY, classValidY, classTestY     = np.split( classY,  [classTrainLimit,classValidLimit] )

classTrainGenPtH, classValidGenPtH, classTestGenPtH  = np.split( classGenPtH,  [classTrainLimit,classValidLimit] )
classTrainMjj, classValidMjj, classTestMjj           = np.split( classMjj,  [classTrainLimit,classValidLimit] )
classTrainGenMjj, classValidGenMjj, classTestGenMjj  = np.split( classGenMjj,  [classTrainLimit,classValidLimit] )
classTrainPtHjj,  classValidPtHjj, classTestPtHjj    = np.split( classPtHjj,  [classTrainLimit,classValidLimit] )
classTrainGenPtHjj,  classValidGenPtHjj, classTestGenPtHjj  = np.split( classGenPtHjj,  [classTrainLimit,classValidLimit] )

#Normalise features and encode target column
y_train_onehot  = np_utils.to_categorical(classTrainJ, num_classes=3)
y_valid_onehot  = np_utils.to_categorical(classValidJ, num_classes=3)
y_test_onehot   = np_utils.to_categorical(classTestJ, num_classes=3)
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(classTrainI)
X_valid_scaled  = scaler.transform(classValidI)
X_test_scaled   = scaler.transform(classTestI)

#(pd.DataFrame(X_test_scaled)).to_pickle('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/debug/X_test_scaled.pkl')

# read in the HPs:
paramExt = ''
trainParams = {}
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]

print(trainParams)

numLayers = int(trainParams['hiddenLayers'],10)
nodes     = int(trainParams['nodes'],10)
dropout   = float(trainParams['dropout'])
batchSize = int(trainParams['batchSize'],10)

#save file
#scalerFile      = ('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/NNscalers/XTrainScalar_nJet__%s.save' % (paramExt))
#joblib.dump(scaler, scalerFile)

#build the jet-classifier NN
num_inputs  = X_train_scaled.shape[1]
num_outputs = nJetClasses

model = Sequential()                                                                                            

for i, nodes in enumerate([200] * numLayers):
  if i == 0: #first layer
    model.add(
    Dense(
            nodes,
            kernel_initializer='glorot_normal',
            activation='relu',
            kernel_regularizer=l2(1e-5),
            input_dim=num_inputs
            )
    )
    model.add(Dropout(dropout))
  else: #hidden layers
    model.add(
    Dense(
            nodes,
            kernel_initializer='glorot_normal',
            activation='relu',
            kernel_regularizer=l2(1e-5),
            )
    )
    model.add(Dropout(dropout))

#final layer
model.add(
        Dense(
            num_outputs,
            kernel_initializer=RandomNormal(),
            activation='softmax'
            )
        )

model.compile(
        loss='categorical_crossentropy',
        optimizer=Nadam(),
        metrics=['accuracy']
)
callbacks = []
callbacks.append(EarlyStopping(patience=50))
model.summary()


print 'about to train jet counting NN'

history = model.fit(
    X_train_scaled,
    y_train_onehot,
    class_weight=classTrainFW,
    validation_data=(X_valid_scaled,y_valid_onehot),
    batch_size=batchSize,
    epochs=1000,
    shuffle=True,
    callbacks=callbacks
    )
print 'done'


#save model - don't do if optimising the model

modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
model.save('%s/nJetNN_MCWeights_%s.h5'%(modelDir,paramExt))
print 'saved as %s/nJetNN_MCWeights_%s.h5'%(modelDir,paramExt)


'''
#get predicted values (priors)
predProbJet = model.predict(testingJet).reshape(classTestJ.shape[0],nJetClasses) #FIXME: same as this but for 9 clases
totSumW = sumW_0J + sumW_1J +sumW_2J 
priors = np.array( [sumW_0J/totSumW, sumW_1J/totSumW, sumW_2J/totSumW] ) 
predProbJet *= priors 
classPredJ = np.argmax(predProbJet, axis=1) 
'''

#Get predicted values (no priors)
yProb      = model.predict(X_test_scaled)
classPredJ = yProb.argmax(axis=-1)

#create dataframe to do pred cats
predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(classTestP), pd.DataFrame(classTestPtHjj), pd.DataFrame(classTestMjj) ], axis=1)
predFrame.columns = ['n_pred_jets', 'diphopt', 'ptHjj', 'dijet_Mjj']
predFrame['predggHClass'] = predFrame.apply(jetPtToggHClass, axis=1)
classPredY = (predFrame['predggHClass'].values).astype('int')


print
print 'reconstructed  number of jets =  %s'%classTestN.astype('int')
print 'NN predicted  number of jets  =  %s'%classPredJ
print 'truth number of jets          =  %s'%classTestJ
print 'NN predicted  number of jets  =  %s'%classPredJ
print 'reconstructed diphoton pt     =  %s'%classTestP
print
print '                   reco class =  %s' %classTestR
print '          NN predicted class  =  %s'%classPredY
print '                  truth class =  %s'%classTestY
print

#Evaluate accuracy score for STXS classes (not jet classes)
print
print('Accuracy score for the NN is: %.4f' % (accuracy_score(classTestY,classPredY,sample_weight=classTestFW)))
print('Accuracy score for Reco is  : %.4f' % (accuracy_score(classTestY,classTestR,sample_weight=classTestFW)))

NNaccuracy = accuracy_score(classTestY,classPredY,sample_weight=classTestFW)

#calculate the log loss
mLogLoss = log_loss(classTestJ, yProb)#, sample_weight=classTestJW)
print 'NN log-loss=  %.4f' %mLogLoss

########
#predict on entire data set, normalising using same standard scalar as before
#classI = scaler.transform(classI)
#y_prob = model.predict(classI)
#classPredJ = y_prob.argmax(axis=-1)
#predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(classP)], axis=1)
#predFrame.columns = ['n_pred_jets', 'diphopt'] 
#predFrame['predggHClass'] = predFrame.apply(jetPtToggHClass, axis=1)
#classPredY = (predFrame['predggHClass'].values).astype('int')
#print('Accuracy score on model evaluated on entire data set is : %.4f' % (accuracy_score(classY,classPredY,sample_weight=classFW)))
#print
########

### Plotting ### NOTE: we dont actually use any plots made here; do the main plotting in data sigs

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
#canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/recoPredJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/recoPredJetHist%s.png'%(plotDir,paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/recoEfficiencyJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/recoEfficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/recoEfficiencyJetHist%s.pdf'%(paramExt))

#print effeciency averages to text file for comparisons (reco) (first 9 bins only)
'''
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

#setup more 1D hists for truth class, NN PREDICTED class, right, and wrong
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
   
#plot and save results for NN PREDICTION 
wrongHist.Add(rightHist)
rightHist.Divide(wrongHist)
effHist = r.TH1F
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/truthJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/truthJetHist%s.png'%(plotDir,paramExt))
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/predJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/predJetHist%s.png'%(plotDir,paramExt))
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist TEXT')
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='%.1f fb^{-1}'%opts.intLumi)
#canv.Print('%s/efficiencyJetHist%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/efficiencyJetHist%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/efficiencyJetHist%s.pdf'%(paramExt))

#print some stat params to text file for comparisons (BDT pred) (first 9 bins only)
'''
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

#print log-loss to text file (used in HP optimiastion). Print other params too for interest                     
'''
lossFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nJetNN/losses_NoW.txt','a+')
lossHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nJetNN/losses_NoW_HP.txt','a+')
eff_array = np.zeros(nClasses)
for iBin in range(1, nClasses+1):
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
print 'loss list is:'
print (eff_array)
print('NN avg eff: %1.4f' %(np.average(eff_array)))
lines = lossFile.readlines()
lossList = []
for line in lines:
  lossList.append(float(line))
if (mLogLoss < lossList[-1]):
  lossFile.write('%1.4f\n' % mLogLoss) 
  lossHPFile.write('HPs: %s --> loss: %1.4f. acc:%1.4f. <eff>: %1.4f\n' % (trainParams,mLogLoss,NNaccuracy,np.average(eff_array)) ) 
lossFile.close()
lossHPFile.close()
'''

'''
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
#canv.Print('%s/procJetHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procJetHistReco%s.png'%(plotDir,paramExt))
prettyHist(catHistReco)
catHistReco.Draw('colz,text')
#canv.Print('%s/catJetHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catJetHistReco%s.png'%(plotDir,paramExt))
prettyHist(procHistPred)
procHistPred.Draw('colz,text')
#canv.Print('%s/procJetHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procJetHistPred%s.png'%(plotDir,paramExt))
prettyHist(catHistPred)
catHistPred.Draw('colz,text')
#canv.Print('%s/catJetHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catJetHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nJetNNPlots/PurityMatrix.pdf')

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
effArrayNN = correctArray/(correctArray+incorrectArray)

#NB this metric does not care about number of events in cats, unlike accuracy
print('Cat effs are:')
print(effArrayNN)
print('\nAverage NN eff is: %f'%effArrayNN.mean())

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
print('Effs are:')
print(effArrayReco)
print('\nAverage Reco eff is: %f'%effArrayReco.mean())

#do plotting
df = pd.DataFrame({
    'group':['NN','Reco'],
    'Bin 1': [effArrayNN[0], effArrayReco[0]],
    'Bin 2': [effArrayNN[1], effArrayReco[1]],
    'Bin 3': [effArrayNN[2], effArrayReco[2]],
    'Bin 4': [effArrayNN[3], effArrayReco[3]],
    'Bin 2': [effArrayNN[1], effArrayReco[1]],
    'Bin 3': [effArrayNN[2], effArrayReco[2]],
    'Bin 4': [effArrayNN[3], effArrayReco[3]],
    'Bin 5': [effArrayNN[4], effArrayReco[4]],
    'Bin 6': [effArrayNN[5], effArrayReco[5]],
    'Bin 7': [effArrayNN[6], effArrayReco[6]],
    'Bin 8': [effArrayNN[7], effArrayReco[7]],
    'Bin 9': [effArrayNN[8], effArrayReco[8]] 
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
#NN
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="NN", color='b')
ax.fill(angles, values, 'b', alpha=0.1)
 
# Reco
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Reco", color='r')
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
#plt.savefig('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/NNRadarPlot.pdf')
'''
