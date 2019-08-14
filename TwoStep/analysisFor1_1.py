import ROOT
import ast
import math
import sys
import os
from array import array
import usefulStyle as useSty

#Input trees
f_input = ROOT.TFile("/vols/cms/jwd18/Stage1categorisation/Pass1/2016/trees/Merged.root")

#Get the tree from the ROOT file
tree = f_input.Get('/vbfTagDumper/trees/ggh_125_13TeV_VBFDiJet')

#Configure output: contains histograms
f_output = ROOT.TFile.Open("hist.root","RECREATE")

#Define histograms and label bins
con_matrix = ROOT.TH2F("confusion matrix","Confusion matrix for reco cats v.s. production bins",13,1,14,23,1,24)
h_norm_con_matrix = ROOT.TH2F("Normalised confusion matrix",";;;Category signal composition (%)",13,1,14,23,1,24)
h_matrix_nonorm = ROOT.TH2F("Unormalised confusion matrix","",13,1,14,23,1,24)
h_stage1p1 = ROOT.TH1F("1p1","",13,1,14)
h_sigs_indiv = ROOT.TH1F("hbarchart","Total ggH bin significance for two optimisation strategies",9,0,9)
h_sigs_comb = ROOT.TH1F("hbarchart","Total ggH bin significance for two optimisation strategies",9,0,9)

#read in text file of optimised cuts, without lumi scaling (only sigs are scaled)
cutFile = open('BDTCutsIndiv.txt','r')
lines = cutFile.readlines()
BDTCuts = []
for line in lines:
  BDTCuts.append(float(line))
print('Indiv BDT Cuts are:')
print(BDTCuts)

combCutFile = open('BDTCutsComb.txt','r')
lines = combCutFile.readlines()
BDTCutsComb = []
for line in lines:
  BDTCutsComb.append(float(line))
print('Comb cuts (not used in purity matrix!) are:')
print(BDTCutsComb)

indivSigFile = open('CatSigsIndiv.txt','r')
lines = indivSigFile.readlines()
indivSigs = []
for line in lines:
  indivSigs.append(float(line))
print('Individual sigs are:')
print(indivSigs)

combSigFile = open('CatSigsComb.txt','r')
lines = combSigFile.readlines()
combSigs = []
for line in lines:
  combSigs.append(float(line))
print('Combined sigs are:')
print(combSigs)

#Vars for looping over events
evtCounter = 0
maxEvents = 10000
totalEntries = tree.GetEntries()
counter = 0
norm_by_row = True 

#Note that the commented bins refer to the sub sub bins 
def binnerOnePointOne(ev):
  
  if(ev.gen_pTH < 200):
    if(ev.n_gen_jets == 0):
      if(ev.gen_pTH < 10): return 1
      else: return 2
    if(ev.n_gen_jets == 1): 
      if(ev.gen_pTH < 60): return 3
      elif(ev.gen_pTH < 120): return 4
      elif(ev.gen_pTH < 200): return 5 
    if(ev.n_gen_jets >= 2): 
      if(ev.gen_dijet_Mjj < 350):
        if(ev.gen_pTH < 60): return 6 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
        elif(ev.gen_pTH < 120): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
        elif(ev.gen_pTH < 200): return 8
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
      else: #( implicit if Mjj>350)
        if(ev.gen_ptHjj < 25): 
          if(ev.gen_dijet_Mjj < 700): return 10
          else: return 11
        else:
          if(ev.gen_dijet_Mjj < 700): return 12
          else: return 13
        #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25):
          #if(ev.gen_dijet_Mjj < 700): return 12
          #if(ev.gen_dijet_Mjj > 700 and ev.gen_dijet_Mjj < 1000): return 13
          #if(ev.gen_dijet_Mjj > 1000 and ev.gen_dijet_Mjj < 1500): return 14
          #if(ev.gen_dijet_Mjj > 1500): return 15
        #if(ev.gen_ptHjj > 25):
          #if(ev.gen_dijet_Mjj < 700): return 16
          #if(ev.gen_dijet_Mjj > 700 and ev.gen_dijet_Mjj < 1000): return 17
          #if(ev.gen_dijet_Mjj > 1000 and ev.gen_dijet_Mjj < 1500): return 18
          #if(ev.gen_dijet_Mjj > 1500): return 19 
  elif(ev.gen_pTH > 200): return 9
  else: return -1 #everything that doesn't go into a bin

  
def categoriserOnePointOne(ev):
  #Stage1.1 reco/cat definitions
  if(ev.dipho_pt < 200):
    if(ev.n_rec_jets == 0):
      if(ev.dipho_pt < 10): 
        if(ev.diphomvaxgb > BDTCuts[0]): return 1
        elif(ev.diphomvaxgb > BDTCuts[1]): return 2   
        else: return 23
      else:
        if(ev.diphomvaxgb > BDTCuts[2]): return 3
        elif(ev.diphomvaxgb > BDTCuts[3]): return 4   
        else: return 23 

    if(ev.n_rec_jets == 1): 
      if(ev.dipho_pt < 60): 
        if(ev.diphomvaxgb > BDTCuts[4]): return 5
        elif(ev.diphomvaxgb > BDTCuts[5]): return 6   
        else: return 23 
      elif(ev.dipho_pt < 120): 
        if(ev.diphomvaxgb > BDTCuts[6]): return 7
        elif(ev.diphomvaxgb > BDTCuts[7]): return 8   
        else: return 23
      elif(ev.dipho_pt < 200):
        if(ev.diphomvaxgb > BDTCuts[8]): return 9
        elif(ev.diphomvaxgb > BDTCuts[9]): return 10   
        else: return 23

    if(ev.n_rec_jets >= 2): 
      if(ev.dijet_Mjj < 350):
        if(ev.dipho_pt < 60): 
          if(ev.diphomvaxgb > BDTCuts[10]): return 11
          elif(ev.diphomvaxgb > BDTCuts[11]): return 12   
          else: return 23
        elif(ev.dipho_pt < 120): 
          if(ev.diphomvaxgb > BDTCuts[12]): return 13
          elif(ev.diphomvaxgb > BDTCuts[13]): return 14  
          else: return 23 
        elif(ev.dipho_pt < 200): 
          if(ev.diphomvaxgb > BDTCuts[14]): return 15
          elif(ev.diphomvaxgb > BDTCuts[15]): return 16   
          else: return 23
      else: #( implicit if Mjj>350). No optimisation of BDT.
        if(ev.ptHjj < 25):
          if(ev.dijet_Mjj < 700): return 19
          else: return 20
        else: #(implicit if PtHjj > 25)
          if(ev.dijet_Mjj < 700): return 21
          else: return 22
  elif(ev.dipho_pt > 200): 
    if(ev.diphomvaxgb > BDTCuts[16]): return 17
    elif(ev.diphomvaxgb > BDTCuts[17]): return 18   
    else: return 23


for ev in tree:

  xBinResult = binnerOnePointOne(ev)
  yCatResult = categoriserOnePointOne(ev)
  con_matrix.Fill(xBinResult,yCatResult,ev.weight)
  h_matrix_nonorm.Fill(xBinResult,yCatResult,ev.weight)
  h_stage1p1.Fill(xBinResult, ev.weight)
 
  if(xBinResult==-1): 
    print 'didnt get an x bin...'

  #evtCounter += 1 
  #if(evtCounter > maxEvents): break  
  
#Need to scale by the integral of each row or column 
xBins = (con_matrix.GetNbinsX())
yBins = (con_matrix.GetNbinsY())
scale = 0


if(norm_by_row==True):
#Add one because Python's range() goes up to one minus max value 
  for yBin in range(1,yBins+1):
    #calculate the integral of given row  
    for xBin in range(1,xBins+1):
      scale += con_matrix.GetBinContent(xBin,yBin)    
    print 'scale is %f' %(scale)
    if (scale == 0): 
      print 'Scale was equal to zero...'
    for xBin in range(1,xBins+1):
      value = con_matrix.GetBinContent(xBin,yBin)
      #Set bins with % less than 0.5 to zero for aesthetic
      if( 100*value/scale < 0.5): 
        h_norm_con_matrix.SetBinContent(xBin,yBin,0)
      else:   
        h_norm_con_matrix.SetBinContent(xBin,yBin,100*value/scale)
    scale = 0
else: #Norm by col
  for xBin in range(1,xBins+1):
    for yBin in range(1,yBins+1):
      scale += con_matrix.GetBinContent(xBin,yBin)    
    print 'scale is %f' %(scale)
    if (scale == 0): 
      'Scale was equal to zero...'
    for yBin in range(1,yBins+1):
      value = con_matrix.GetBinContent(xBin,yBin)
      if( 100*value/scale < 0.5): 
        h_norm_con_matrix.SetBinContent(xBin,yBin,0)
      else:   
        h_norm_con_matrix.SetBinContent(xBin,yBin,100*value/scale)
    scale = 0
    

#Set up color pallete
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
ROOT.TColor.CreateGradientColorTable(npoints, stops, red, green, blue, ncontours, alpha)

#set histogram style options
ROOT.gStyle.SetPaintTextFormat("4.0f")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetNumberContours(ncontours)
#ROOT.gStyle.SetGridStyle(1)
#ROOT.gStyle.SetLineWidth(1)
ROOT.gStyle.SetGridColor(16)

# set up canvas
c1 = ROOT.TCanvas("c1","c1",1000,700)
c1.SetRightMargin(0.15)
c1.SetLeftMargin(0.25)
c1.SetBottomMargin(0.22)
c1.SetGridy()

#Draw purity matrix (un-normalised)
h_matrix_nonorm.GetXaxis().SetTitle("STXS process")
h_matrix_nonorm.GetYaxis().SetTitle("Reco category")
h_matrix_nonorm.GetYaxis().SetTitleOffset(4)
h_matrix_nonorm.GetXaxis().SetTitleOffset(2.5)

h_matrix_nonorm.GetXaxis().SetBinLabel(1,"ggH 0J low p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(2,"ggH 0J med p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(3,"ggH 1J low p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(4,"ggH 1J med p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(5,"ggH 1J high p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(6,"ggH 2J low mjj low p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(7,"ggH 2J low mjj med p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(8,"ggH 2J low mjj high p_{T}^{H}")
h_matrix_nonorm.GetXaxis().SetBinLabel(9,"BSM")
h_matrix_nonorm.GetXaxis().SetBinLabel(10,"ggH 2J h l l")
h_matrix_nonorm.GetXaxis().SetBinLabel(11,"ggH 2J h l h")
h_matrix_nonorm.GetXaxis().SetBinLabel(12,"ggH 3J h h l")
h_matrix_nonorm.GetXaxis().SetBinLabel(13,"ggH 3J h h h")
h_matrix_nonorm.GetXaxis().LabelsOption("v")

h_matrix_nonorm.GetYaxis().SetBinLabel(1,"0J low Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(2,"0J low Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(3,"0J med Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(4,"0J med Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(5,"1J low Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(6,"1J low Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(7,"1J med Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(8,"1J med Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(9,"1J high Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(10,"1J high Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(11,"2J low mjj low Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(12,"2J low mjj low Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(13,"2J low mjj med Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(14,"2J low mjj med Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(15,"2J low mjj high Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(16,"2J low mjj high Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(17,"BSM Tag 0")
h_matrix_nonorm.GetYaxis().SetBinLabel(18,"BSM Tag 1")
h_matrix_nonorm.GetYaxis().SetBinLabel(19,"2J h l l")
h_matrix_nonorm.GetYaxis().SetBinLabel(20,"2J h l h")
h_matrix_nonorm.GetYaxis().SetBinLabel(21,"3J h h l")
h_matrix_nonorm.GetYaxis().SetBinLabel(22,"3J h h h")
h_matrix_nonorm.GetYaxis().SetBinLabel(23,"Other")

h_matrix_nonorm.Draw("COLZ TEXT")

#Draw lines separating ggh and VBF-like bins and cats
h_line = ROOT.TLine(1,19,14,19)
h_line.SetLineColorAlpha(ROOT.kGray,0.8)
h_line.Draw()
v_line = ROOT.TLine(10,1,10,24)
_line.SetLineColorAlpha(ROOT.kGray,0.8)
v_line.Draw()
h_line_fix1 = ROOT.TLine(1,24,14,24)
h_line_fix1.Draw()
h_line_fix2 = ROOT.TLine(1,1,14,1)
h_line_fix2.Draw()
v_line_fix = ROOT.TLine(19,1,19,24)
v_line_fix.Draw()


#Draw and label normalised purity matrix
c2 = ROOT.TCanvas("c2","c2",1200,750)
c2.SetRightMargin(0.15)
c2.SetLeftMargin(0.25)
c2.SetBottomMargin(0.27)
c2.SetGridy()

h_norm_con_matrix.GetXaxis().SetTitle("STXS process")
h_norm_con_matrix.GetYaxis().SetTitle("Reco category")
h_norm_con_matrix.GetYaxis().SetTitleOffset(3.1)
h_norm_con_matrix.GetXaxis().SetTitleOffset(4.05)
h_norm_con_matrix.GetZaxis().SetTitleOffset(1.2)
h_norm_con_matrix.GetYaxis().SetTickLength(0)
h_norm_con_matrix.GetXaxis().SetTickLength(0)

h_norm_con_matrix.GetXaxis().SetBinLabel(1,"0J low p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(2,"0J med p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(3,"1J low p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(4,"1J med p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(5,"1J high p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(6,"2J low mjj low p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(7,"2J low mjj med p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(8,"2J low mjj high p^{_{H}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(9,"BSM")
h_norm_con_matrix.GetXaxis().SetBinLabel(10,"2J med mjj low p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(11,"2J med mjj high p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(12,"2J high mjj low p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetXaxis().SetBinLabel(13,"2J high mjj high p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetXaxis().LabelsOption("v")

h_norm_con_matrix.GetYaxis().SetBinLabel(1,"0J low p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(2,"0J low p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(3,"0J med p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(4,"0J med p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(5,"1J low p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(6,"1J low p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(7,"1J med p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(8,"1J med p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(9,"1J high p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(10,"1J high p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(11,"2J low mjj low p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(12,"2J low mjj low p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(13,"2J low mjj med p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(14,"2J low mjj med p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(15,"2J low mjj high p^{_{H}}_{^{T}} Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(16,"2J low mjj high p^{_{H}}_{^{T}} Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(17,"BSM Tag 0")
h_norm_con_matrix.GetYaxis().SetBinLabel(18,"BSM Tag 1")
h_norm_con_matrix.GetYaxis().SetBinLabel(19,"2J med mjj low p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetYaxis().SetBinLabel(20,"2J med mjj high p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetYaxis().SetBinLabel(21,"2J high mjj low p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetYaxis().SetBinLabel(22,"2J high mjj high p^{_{Hjj}}_{^{T}}")
h_norm_con_matrix.GetYaxis().SetBinLabel(23,"Other")

h_norm_con_matrix.SetMaximum(100)
h_norm_con_matrix.Draw("COLZ TEXT")
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='35.9 fb^{-1} (2016)')

#Draw lines separating ggh and VBF-like bins and cats
h_line.Draw()
v_line.Draw()
h_line_fix1.Draw()
h_line_fix2.Draw()
v_line_fix.Draw()

#Make bar chart for the significance comparison
c3 = ROOT.TCanvas("c3","c3",800,500)
c3.SetGrid()

for i,sig in enumerate(indivSigs):
  h_sigs_indiv.SetBinContent(i+1,sig)
  h_sigs_indiv.SetFillColor(4)
  h_sigs_indiv.SetBarWidth(0.4)
  h_sigs_indiv.SetBarOffset(0.1)
  h_sigs_indiv.GetXaxis().SetBinLabel(i+1,'Bin %.0f'%(i+1))

for i,sig in enumerate(combSigs):
  h_sigs_comb.SetBinContent(i+1,sig)
  h_sigs_comb.SetFillColor(38)
  h_sigs_comb.SetBarWidth(0.4)
  h_sigs_comb.SetBarOffset(0.5)

h_sigs_indiv.Draw("BAR")
h_sigs_indiv.GetYaxis().SetTitle("Significance")
h_sigs_comb.Draw("BAR SAME")

bar_chart_legend = ROOT.TLegend(0.55,0.7,0.77,0.81)
bar_chart_legend.AddEntry(h_sigs_indiv,"Individual Opt","f") 
bar_chart_legend.AddEntry(h_sigs_comb,"Combined Opt","f") 
bar_chart_legend.Draw()

#save canvases
c1.SaveAs("PurityMatrixStage1Point1_NoNorm.pdf")
c1.SaveAs("PurityMatrixStage1Point1_NoNorm.png")

if(norm_by_row==True):
  c2.SaveAs("PurityMatrixStage1Point1_with_BDT.pdf")
  c2.SaveAs("PurityMatrixStage1Point1_with_BDT.png")
else:
  c2.SaveAs("PurityMatrixStage1Point1_with_BDT_norm_by_col.pdf")
  c2.SaveAs("PurityMatrixStage1Point1_with_BDT_norm_by_col.png")

c3.SaveAs("SigComparison.pdf")
c3.SaveAs("SigComparison.png")

#Plot stage 1p1 histogram (gen level)
c4 = ROOT.TCanvas("c4","c4",800,500)
h_stage1p1.Draw("hist")
useSty.drawCMS(onTop=True)
useSty.drawEnPu(lumi='35.9 fb^{-1} (2016)')
h_stage1p1.GetXaxis().SetTitle("STXS process")

raw_input("press any key to continue")
#Close output file
f_output.Write()
f_output.Close()
