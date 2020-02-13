#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
import uproot as upr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthClass1p2, jetPtToggHClass, truthClass, applyLumiScale, ptSplits
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma, jetPtToClass 
from root_numpy import tree2array, fill_hist
from catOptim import CatOptim
from math import pi
import usefulStyle as useSty

from variableDefinitions import allVarsGen

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signifFrame', default=None, help='Name of cleaned signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('-c','--className', default=None, help='Name of multi-class model used to build categories. If None, use reco categories')
parser.add_option('-n','--nIterations', default=10, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
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
nClasses = 9
catNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high','200<BSM<300', '300<BSM<450','450<BSM<650', '650<BSM']
binNames = ['0J low','0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high', '200<BSM<300', '300<BSM<450','450<BSM<650','650<BSM','VBF-like']
jetPriors = [0.606560, 0.270464, 0.122976]
procPriors = [0.146457, 0.517408, 0.151512, 0.081834, 0.014406, 0.020672, 0.036236, 0.017924, 0.013551]

yearToLumi = {'2016':35.9, '2017':41.5, '2018':59.7}

#put root in batch mode
r.gROOT.SetBatch(True)

signals = ['ggh']
backgrounds = ['Data']

#define the different sets of variables used
diphoVars  = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
               'dipho_leadEta','dipho_subleadEta',
               'CosPhi','vtxprob','sigmarv','sigmawv']

varsForCuts = ['dipho_leadIDMVA', 'dipho_mass', 'dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
               'dipho_leadEta','dipho_subleadEta', 'dijet_Mjj', 'dipho_PToM', 'dipho_pt',
               'CosPhi','vtxprob','sigmarv','sigmawv', 'weight', 'n_jet_30', 'dipho_dijet_ptHjj', 'HTXSstage1p2bin']
years       = ['year_2016', 'year_2017', 'year_2018']

jetVars    = ['n_jet_30','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubleadJPt','dijet_SubsubleadJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

 
allVars   = ['n_jet_30','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubleadJPt','dijet_SubsubleadJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'CosPhi','vtxprob','sigmarv','sigmawv','dipho_mass', 'weight', 'dipho_PToM', 'dipho_dijet_ptHjj',                'HTXSstage1p2bin']

queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.9) and (dipho_subleadIDMVA>-0.9) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_Mjj<350.)'

#m,procFileMap = {'Data':'Data_jetInfo.root'} 
#procFileMap = {'Data':'Data_2016.root','Data':'Data_2017.root','Data':'Data_2018.root'}
procFileMap = {'Data':'Data_combined.root'} 
theProcs = procFileMap.keys()

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile = upr.open('%s/%s'%(trainDir,fn)) 
      if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      trainFrames[proc] = trainTree.pandas.df(varsForCuts[:-1]+jetVars).query(queryString)
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  dataTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'


  cutFlow2017 = open('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/debug/cutflow/cutFlow2017_data.txt','w+')
  cutFlow2017.write('Start with this many signal events: ')
  totEvents = np.sum(dataTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % totEvents) 

  cutFlow2017.write('Following mass cuts, we have: ')
  afterMassCut = np.sum(dataTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % afterMassCut)
  effRemovedByCut = (totEvents - afterMassCut)/totEvents
  cutFlow2017.write('So cut removed: %.3f of total events \n' %(effRemovedByCut)) 

  cutFlow2017.write('Following preselection cuts, we finally have: ')
  afterPreselCut = np.sum(dataTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % afterPreselCut)
  effRemovedByCut = (afterMassCut-afterPreselCut)/totEvents
  cutFlow2017.write('So cut removed: %.3f of total events \n' %(effRemovedByCut)) 
  cutFlow2017.close()
  
  #add extra info to dataframe
  print 'about to add extra columns'
  dataTotal['diphopt'] = dataTotal.apply(addPt, axis=1)
  dataTotal['reco'] = dataTotal.apply(reco, axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_hdf('%s/dataTotal.h5'%frameDir, key='df', mode='w', format='table')
  print 'frame saved as %s/dataTotal.h5'%frameDir

#read in dataframe if above steps done before
else:
  dataTotal = pd.read_hdf('%s/%s'%(frameDir,opts.dataFrame), 'df')
  print 'Successfully loaded the dataframe'

print('dataTotal:')
print(dataTotal['weight'].head(30))

if not opts.signifFrame:
  #sigFileMap = {'ggh':'ggH_amc_jetinfo.root'}
  #sigFileMap = {'ggh':'ggh_amc_withBinaryYear_2016.root', 'ggh':'ggh_amc_withBinaryYear_2017.root', 'ggh':'ggh_amc_withBinaryYear_2018.root'} 
  sigFileMap = {'ggh':'ggH_amc_combined.root'} 
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in sigFileMap.iteritems():
      trainFile = upr.open('%s/%s'%(trainDir,fn)) 
      if proc in signals: trainTree = trainFile['%s_125_13TeV_GeneralDipho'%proc]
      #if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      trainFrames[proc] = trainTree.pandas.df(allVars+years).query(queryString)
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc,fn in sigFileMap.iteritems():
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  cutFlow2017 = open('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/debug/cutflow/cutFlow2017_sig.txt','w+')
  cutFlow2017.write('Start with this many signal events: ')
  totEvents = opts.intLumi*np.sum(trainTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % totEvents) 
  

  #then filter out the events into only those with the phase space we are interested in

  cutFlow2017.write('Following mass cuts, we have: ')
  afterMassCut = opts.intLumi*np.sum(trainTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % afterMassCut)
  effRemovedByCut = (totEvents - afterMassCut)/totEvents
  cutFlow2017.write('So cut removed: %.3f of total events \n' %(effRemovedByCut)) 

  #FIXME below is temporarily replaced
  #read in signal mc dataframe
  #trainTotal = pd.read_pickle('%s/trainTotal.pkl'%frameDir)
  #trainTotal = pd.read_pickle('%s/jetTotal.pkl'%frameDir)
  #print 'Successfully loaded the signal dataframe'

  print 'About to add reco tag info'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho, axis=1, args=[signals])
  #trainTotal['truthClass'] = trainTotal.apply(truthClass, axis=1)
  trainTotal['truthClass'] = trainTotal.apply(truthClass1p2, axis=1)
  print 'Successfully added reco tag info'

  cutFlow2017.write('Following preselection cuts, we have: ')
  afterPreselCut = opts.intLumi*np.sum(trainTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % afterPreselCut)
  effRemovedByCut = (afterMassCut-afterPreselCut)/totEvents
  cutFlow2017.write('So cut removed: %.3f of total events \n' %(effRemovedByCut)) 
  
  trainTotal = trainTotal[trainTotal.reco>-1]
  trainTotal = trainTotal[trainTotal.truthClass>-1]
 
  cutFlow2017.write('Following truth class cuts, we finally have: ')
  afterTruthCut = opts.intLumi*np.sum(trainTotal['weight'].values)
  cutFlow2017.write('%.3f \n' % afterTruthCut)
  effRemovedByCut = (afterPreselCut-afterTruthCut)/totEvents
  cutFlow2017.write('So cut removed: %.3f of total events \n' %(effRemovedByCut)) 

  cutFlow2017.close()

  #save
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_hdf('%s/signifTotal.h5'%frameDir, key='df', mode='w', format='table')
  print 'frame saved as %s/signifTotal.h5'%frameDir


else:
  #read in already cleaned up signal frame
  trainTotal = pd.read_hdf('%s/%s'%(frameDir,opts.signifFrame), 'df')

trainTotal['weight'] = trainTotal.apply(applyLumiScale, axis=1, args=[yearToLumi])

print('TrainTotal:')
print(trainTotal['weight'].head(30))

#define the variables used as input to the classifier
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
if opts.className:
  if 'Jet' in opts.className:
    diphoI  = trainTotal[jetVars].values
  if 'Class' in opts.className:
    diphoI  = trainTotal[jetVars+diphoVars+['diphopt']].values
diphoJ  = trainTotal['truthClass'].values
diphoFW = trainTotal['weight'].values
diphoP  = trainTotal['diphopt'].values
diphoR  = trainTotal['reco'].values
diphoM  = trainTotal['dipho_mass'].values

dataX  = dataTotal[diphoVars].values
if opts.className:
  if 'Jet' in opts.className:
    dataI  = dataTotal[jetVars].values
  if 'Class' in opts.className:
    dataI  = dataTotal[jetVars+diphoVars+['diphopt']].values
dataY  = np.zeros(dataX.shape[0])
dataFW = np.ones(dataX.shape[0])
dataP  = dataTotal['diphopt'].values
dataR  = dataTotal['reco'].values
dataM  = dataTotal['dipho_mass'].values

#setup matrices
diphoMatrix = xg.DMatrix(diphoX, label=diphoY, weight=diphoFW, feature_names=diphoVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataY,  weight=dataFW,  feature_names=diphoVars)

#load the dipho model to be tested
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))

#get predicted values
diphoPredY = diphoModel.predict(diphoMatrix)
dataPredY  = diphoModel.predict(dataMatrix)

print('Reco accuracy score is:')
print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))

#load the class model to be tested, if it exists
if opts.className:
  classModel = xg.Booster()
  classModel.load_model('%s/%s'%(modelDir,opts.className))

  if 'Jet' in opts.className:
    classMatrix = xg.DMatrix(diphoI, label=diphoY, weight=diphoFW, feature_names=jetVars)
    predProbJet = classModel.predict(classMatrix).reshape(diphoX.shape[0],nJetClasses)
    #predProbJet *= jetPriors
    classPredJ = np.argmax(predProbJet, axis=1)
    #concat the nJet prediction with everything else you need to make class predction
    predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(diphoP)], axis=1)
    predFrame.columns = ['n_pred_jets', 'diphopt']
    predFrame['predClass'] = predFrame.apply(jetPtToggHClass, axis=1)
    diphoR = predFrame['predClass'].values 

    print('nJet BDT accuracy score is:')
    print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))
    dfForDiphoBDTPurityMatrices = pd.concat([pd.DataFrame(diphoR), pd.DataFrame(diphoJ), pd.DataFrame(diphoFW), pd.DataFrame(diphoPredY)], axis=1)
    dfForDiphoBDTPurityMatrices.to_pickle('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/picklesForPurity/nJetBDT_MCW.pkl')

    classDataMatrix = xg.DMatrix(dataI, label=dataY, weight=dataFW, feature_names=jetVars)
    predProbJet = classModel.predict(classDataMatrix).reshape(dataX.shape[0],nJetClasses)
    #predProbJet *= jetPriors
    classPredJ = np.argmax(predProbJet, axis=1)
    predFrame = pd.concat([pd.DataFrame(classPredJ), pd.DataFrame(dataP)], axis=1)
    predFrame.columns = ['n_pred_jets', 'diphopt']
    predFrame['predClass'] = predFrame.apply(jetPtToggHClass, axis=1)
    dataR = predFrame['predClass'].values

  elif 'nClasses' in opts.className:
    classMatrix = xg.DMatrix(diphoI, label=diphoY, weight=diphoFW, feature_names=jetVars+diphoVars+['diphopt'])
    predProbClass = classModel.predict(classMatrix).reshape(diphoI.shape[0],nClasses)
    #For equal weights only:
    #predProbClass *= procPriors
    
    diphoR = np.argmax(predProbClass, axis=1)
    diphoResults = pd.concat([pd.DataFrame(diphoR),pd.DataFrame(diphoP), pd.DataFrame(diphoJ), pd.DataFrame(diphoFW)], axis=1)
    diphoResults.columns = ['diphoR','diphoP', 'diphoJ', 'diphoFW']
    diphoResults['diphoR'] = diphoResults.apply(ptSplits, axis=1)
    diphoR = diphoResults['diphoR'].values
    diphoFW = diphoResults['diphoFW']
    print('nClasses BDT accuracy score:')
    print(accuracy_score(diphoJ, diphoR, sample_weight=diphoFW))
    print('dipho results')
    print(diphoResults.head(50))
    
    #same thing for background (no need to save for purits matrices though)
    classDataMatrix = xg.DMatrix(dataI, label=dataY, weight=dataFW, feature_names=jetVars+diphoVars+['diphopt'])
    predProbClass = classModel.predict(classDataMatrix).reshape(dataI.shape[0],nClasses)
    #predProbClass *= procPriors
    dataR = np.argmax(predProbClass, axis=1)
    dataResults = pd.concat([pd.DataFrame(dataR), pd.DataFrame(dataP)], axis=1)
    dataResults.columns = ['diphoR','diphoP']
    dataResults['diphoR'] = diphoResults.apply(ptSplits, axis=1)
    dataR = dataResults['diphoR'].values

  else:
    raise Exception("your class model type is not yet supported, sorry")


#now estimate two-class significance
#set up parameters for the optimiser
ranges = [ [0.5,1.] ]
names  = ['DiphotonBDT']
printStr = ''
binsRequiringThree = [0,1,2,3,4,5,6,7]
binsRequiringTwo = [8,9,10,11]


plotDir  = '%s/%s/Proc_0'%(plotDir,opts.modelName.replace('.model',''))
if not path.isdir(plotDir): 
    system('mkdir -p %s'%plotDir)

for iClass in binsRequiringThree:
    #if (iClass not in binsRequiringTwo):
        sigWeights = diphoFW * (diphoJ==iClass) * (diphoR==iClass)
        bkgWeights = dataFW * (dataR==iClass)
        optimiser = CatOptim(sigWeights, diphoM, [diphoPredY], bkgWeights, dataM, [dataPredY], 3, ranges, names)
        #optimiser.setTransform(True)
        #optimiser.setConstBkg(True)
        optimiser.optimise(1, opts.nIterations)
        plotDir  = plotDir.replace('Proc_%g'%(iClass-1),'Proc_%g'%iClass)
        if not path.isdir(plotDir): 
            system('mkdir -p %s'%plotDir)
        #optimiser.crossCheck(opts.intLumi,plotDir)
        printStr += 'Results for bin %g : \n'%iClass
        printStr += optimiser.getPrintableResult()


for iClass in binsRequiringTwo:
    sigWeights = diphoFW * (diphoJ==iClass) * (diphoR==iClass) 
    bkgWeights = dataFW * (dataR==iClass)
    optimiser = CatOptim(sigWeights, diphoM, [diphoPredY], bkgWeights, dataM, [dataPredY], 2, ranges, names)
    #optimiser.setTransform(True)
    #optimiser.setConstBkg(True)
    optimiser.optimise(1, opts.nIterations)
    printStr += 'Results for bin %g : \n'%iClass
    printStr += optimiser.getPrintableResult()





print
print printStr

                                              ### plotting ###
'''
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
catHistReco.SetMarkerSize(1.65)
catHistReco.SetMaximum(100)
catHistReco.SetMinimum(0)
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='2017')
canv.Print('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiClassifierPlots/nClassBDT_MCW_PurityMatrix.pdf')
canv.Print('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiClassifierPlots/nClassBDT_MCW_PurityMatrix.png')

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
procHistReco.SetMarkerSize(1.65)
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='2017')
canv.Print('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiClassifierPlots/nClassBDT_MCW_PurityMatrix_NormByProc.pdf')
canv.Print('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiClassifierPlots/nClassBDT_MCW_PurityMatrix_NormByProc.png')

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

categories = ['0J low', '0J high', '1J low', '1J med', '1J high', '2J low', '2J med', '2J high', 'BSM']
N = len(categories)
 
 
# Calculate angle of axis (plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
#Add text box for axis label
plt.text(6, 0.35, 'CCF', color="grey")
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='black', size=12)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0,0.2,0.4,0.6,0.8,1], ["0.0", "0.2","0.4","0.6","0.8",""], color="black", size=10)
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
plt.savefig('/vols/build/cms/jwd18/NewTwoStep/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/MultiClassifierPlots/nClassBDT_MCW_effRadarPlot.pdf')
'''
