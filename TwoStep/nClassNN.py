#usual imports
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
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass, jetPtToClass, procWeight, sqrtProcWeight, cbrtProcWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from math import pi

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
import h5py

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
plotDir  = checkDir(trainDir.replace('trees','plots') + '/NNCategorisation')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

trainFrac   = 0.7
validFrac   = 0.1
sampleFrac = 1.0
nGGHClasses = 9
sampleFrame = False
equaliseWeights = False
weightScale = True #multiply weight by 1000

binNames = ['0J low','0J high','1J low','1J med','1J high','2J low low','2J low med','2J low high','BSM'] 

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

allVars   = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

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
  print 'all columns added'

  #only select processes relevant for nJet training
  trainTotal = trainTotal[trainTotal.truthJets>-1]
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal = trainTotal[trainTotal.truthClass!=-1]

  #remove vbf_like procs - don't care about predicting these/ dont want to predict these
  #Then apply the weights to the 9 cats
  trainTotal = trainTotal[trainTotal.truthClass<9]
  #trainTotal['procWeight'] = trainTotal.apply(procWeight, axis=1)
  trainTotal['procWeight'] = trainTotal.apply(cbrtProcWeight, axis=1)
  print 'done basic preselection cuts'

  #replace missing entries with -10 to avoid bias from -999 
  trainTotal = trainTotal.replace(-999,-10) 

  # do this step if later reading df with python 2 
  #trainTotal.loc[:, 'proc'] = trainTotal['proc'].astype(str)  
 
 #scale weights or normalised weights closer to one if desired

  if weightScale:
    print('MC weights before were:')
    print(trainTotal['weight'].head(10))
    trainTotal.loc[:,'weight'] *= 1000
    print('weights after scaling are:') 
    print(trainTotal['weight'].head(10))                                                                        

  #save as a pickle file
  trainTotal.to_pickle('%s/multiClassTotal.pkl'%frameDir)
  print 'frame saved as %s/multiClassTotal.pkl'%frameDir

#read in dataframe if above steps done once before
else:
  trainTotal = pd.read_pickle(opts.dataFrame)
  print 'Successfully loaded the dataframe \n'


#Sample dataframe randomly, if we want to test computation time
if sampleFrame:
  trainTotal = trainTotal.sample(frac=sampleFrac, random_state=1)

#used when setting weights for normalisation, in row functions
procWeightDict = {}
for iProc in range(nGGHClasses):  #from zero to 8 are ggH bins
  sumW = np.sum(trainTotal[trainTotal.truthClass==iProc]['weight'].values)
  sumW_proc = np.sum(trainTotal[trainTotal.truthClass==iProc]['procWeight'].values)
  procWeightDict[iProc] = sumW
  print 'Sum of weights for ggH STXS bin %i is: %.2f' %  (iProc,sumW_proc)  
  print 'Frac is %.6f' % (sumW/ (np.sum(trainTotal['weight'].values)))
  print 'Sum of proc weights for bin %i is: %.5f' % (iProc,sumW_proc)

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)
classValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for multiclass training
classI        = trainTotal[allVars].values
classProcW    = trainTotal['procWeight'].values
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
X_train, X_valid, X_test              = np.split( classI,     [classTrainLimit,classValidLimit] )
w_mc_train, w_mc_valid, w_mc_test     = np.split( classFW,    [classTrainLimit,classValidLimit] )
procW_train, procW_valid, procW_test  = np.split( classProcW, [classTrainLimit,classValidLimit] )
classTrainR, classValidR, classTestR  = np.split( classR,     [classTrainLimit,classValidLimit] )
y_train, y_valid, y_test              = np.split( classY,     [classTrainLimit,classValidLimit] )

#one hot encode target column (necessary for keras) and scale training
y_train_onehot  = np_utils.to_categorical(y_train, num_classes=9)
y_valid_onehot  = np_utils.to_categorical(y_valid, num_classes=9)
y_test_onehot   = np_utils.to_categorical(y_test, num_classes=9)
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_valid_scaled  = scaler.fit_transform(X_valid)
X_test_scaled   = scaler.transform(X_test)

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

print 'training HPs:'
print(trainParams)

numLayers = int(trainParams['hiddenLayers'],10)
nodes     = int(trainParams['nodes'],10)
dropout   = float(trainParams['dropout'])
batchSize = int(trainParams['batchSize'],10)

#build the category classifier 
num_inputs  = X_train_scaled.shape[1]
num_outputs = nGGHClasses

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


#Fit the model with best n_trees/estimators
print('Fitting on the training data')
history = model.fit(
    X_train_scaled,
    y_train_onehot,
    sample_weight=w_mc_train,
    validation_data=(X_valid_scaled,y_valid_onehot, w_mc_valid),
    batch_size=batchSize,
    epochs=1000,
    shuffle=True,
    callbacks=callbacks # add function to print stuff out there
    )
print('Done')


#save model

#modelDir = trainDir.replace('trees','models')
#if not path.isdir(modelDir):
#  system('mkdir -p %s'%modelDir)
#model.save('%s/nClassesNNMCweights__%s.h5'%(modelDir,paramExt))
#print 'saved NN as %s/nClassesNNMCWeights__%s.h5'%(modelDir,paramExt)


#plot train and validation acc over time
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig('NNAccuracyHist.pdf')
plt.savefig('NNAccuracyHist.png')
'''
'''
#save model
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
model.save_weights('%s/ggHNeuralNet.h5'%(modelDir))
print 'saved as %s/ggHNeuralNet%s.h5'%(modelDir,paramExt)
'''

'''
#Evaluate performance with priors 
yProb = model.predict(X_test_scaled)
predProbClass = y_prob.reshape(y_test.shape[0],nGGHClasses)
totSumW =  np.sum(trainTotal['weight'].values)
priors = [] #easier to append to list than numpy array. Then just convert after
for i in range(nGGHClasses):
  priors.append(procWeightDict[i]/totSumW)
predProbClass *= np.asarray(priors) #this is part where include class frac, not MC frac
classPredY = np.argmax(predProbClass, axis=1) 
print 'Accuracy score with priors'
print(accuracy_score(y_test, classPredY, sample_weight=w_mc_test))
'''

#Evaluate performance, no priors
y_prob = model.predict(X_test_scaled) 
y_pred = y_prob.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_test, y_pred, sample_weight=procW_test)
print(NNaccuracy)


print
print '                   reco class =  %s' %classTestR
print '          NN predicted class  =  %s'%y_pred
print '                  truth class =  %s'%y_test
print '         Reco accuracy score  =  %.4f' %accuracy_score(y_test,classTestR, sample_weight=w_mc_test) #include orig MC weights here
print 'NN accuracy score (no priors )=  %.4f' %NNaccuracy #include orig MC weights here

mLogLoss = log_loss(y_test, y_prob, sample_weight=procW_test)
print 'NN log-loss=  %.4f' %mLogLoss

### Plotting Purity Matrices ###
classTestFW = w_mc_test

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
for true,guess,w in zip(y_test,classTestR,classTestFW):
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


#print effeciency averages to text file for comparisons (reco) (first 9 bins only)
'''
recoAveragesFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/reco_eff_averages.txt','a+')
recoAveragesHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/reco_eff_averages_HP.txt','a+')
eff_array = np.zeros(nGGHClasses)
for iBin in range(1, nGGHClasses+1):
  #print 'Bin %g: %f'%(iBin, rightHist.GetBinContent(iBin))
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
lines = recoAveragesFile.readlines()
recoAverages = []
for line in lines:
  recoAverages.append(float(line))
if (np.average(eff_array) > recoAverages[-1]):
  recoAveragesFile.write('%1.3f\n' % np.average(eff_array)) 
  recoAveragesHPFile.write('HPs: %s --> avg eff: %1.3f \n' % (trainParams, np.average(eff_array)) ) 
print 'RECO average efficiency for sample frac: %1.2f , is: %1.3f\n' % (sampleFrac, np.average(eff_array))
recoAveragesFile.close()
'''

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

for true,guess,w in zip(y_test,y_pred,classTestFW):
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


#print some stat params to text file for comparisons (BDT pred) (first 9 bins only)
'''
predAveragesFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/pred_eff_averages.txt','a+')
predAveragesHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/pred_eff_averages_HP.txt','a+')
eff_array = np.zeros(nGGHClasses)
for iBin in range(1, nGGHClasses+1):
  #print 'Bin %g: %f'%(iBin, rightHist.GetBinContent(iBin))
  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
lines = predAveragesFile.readlines()
predAverages = []
for line in lines:
  predAverages.append(float(line))
if (np.average(eff_array) > predAverages[-1]):
  predAveragesFile.write('%1.3f\n' % np.average(eff_array)) 
  predAveragesHPFile.write('HPs: %s --> avg eff: %1.3f \n' % (trainParams, np.average(eff_array)) ) 
print 'NN average efficiency for sample frac: %1.2f , is: %1.3f\n' % (sampleFrac, np.average(eff_array))
predAveragesFile.close()
predAveragesHPFile.close()
'''

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
    sumwProcMap[iProc] = np.sum(classTestFW*(y_test==iProc))
    for jProc in range(nGGHClasses):
        sumwProcCatMapPred[(iProc,jProc)] = np.sum(classTestFW*(y_test==iProc)*(y_pred==jProc))
        sumwProcCatMapReco[(iProc,jProc)] = np.sum(classTestFW*(y_test==iProc)*(classTestR==jProc))


#Sum weights for entire predicted catgeory i.e. row for BDT pred cat and Reco cat
sumwCatMapReco = {}
sumwCatMapPred = {}
for iProc in range(nGGHClasses):
    sumwCatMapPred[iProc] = np.sum(classTestFW*(y_pred==iProc))
    sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc))
    #sumwCatMapPred[iProc] = np.sum(classTestFW*(classTred==iProc)*(y_test!=0)) #don't count bkg here
    #sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc)*(y_test!=0))

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
#canv.Print('%s/procNNHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procNNHistReco%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/procJetHistReco%s.pdf'%(paramExt))
prettyHist(catHistReco)
catHistReco.Draw('colz,text')
#canv.Print('%s/catNNHistReco%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catNNHistReco%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/catJetHistReco%s.pdf'%(paramExt))
prettyHist(procHistPred)
procHistPred.Draw('colz,text')
#canv.Print('%s/procNNHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/procNNHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiCatPlots/procJetHistPred%s.pdf'%(paramExt))
prettyHist(catHistPred)
catHistPred.Draw('colz,text')
#canv.Print('%s/catNNHistPred%s.pdf'%(plotDir,paramExt))
#canv.Print('%s/catNNHistPred%s.png'%(plotDir,paramExt))
#canv.Print('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/NNPurityMatrix.pdf')

#print log-loss to text file (used in HP optimiastion). Print other params too for interest                     
#lossFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nClassNN/losses_NoW.txt','a+')
#lossHPFile=open('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Losses/nClassNN/losses_NoW_HP.txt','a+')
#eff_array = np.zeros(nGGHClasses)
#for iBin in range(1, nGGHClasses+1):
#  eff_array[iBin-1] = rightHist.GetBinContent(iBin)
#print 'loss list is:'
#print (eff_array)
#print('NN avg eff: %1.4f' %(np.average(eff_array)))
#lines = lossFile.readlines()
#lossList = []
#for line in lines:
#  lossList.append(float(line))
#if (mLogLoss < lossList[-1]):
#  lossFile.write('%1.4f\n' % mLogLoss) 
#  lossHPFile.write('HPs: %s --> loss: %1.4f. acc:%1.4f. <eff>: %1.4f\n' % (trainParams,mLogLoss,NNaccuracy,np.average(eff_array)) ) 
#lossFile.close()
#lossHPFile.close()

                                   ##### Efficiency plots #####

classTestY  = y_test
classPredY  = y_pred
classTestFW = w_mc_test

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
plt.legend(loc='upper right', bbox_to_anchor=(0.17, 0.04))
plt.savefig('/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/nClassOutputs/NN/MCW/efficinciesRadarPlot.pdf')    
