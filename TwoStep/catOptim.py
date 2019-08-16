from collections import OrderedDict as od
import numpy as np
import ROOT as r
import json
r.gROOT.SetBatch(True)
from root_numpy import fill_hist
import usefulStyle as useSty

plotDir = '/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/debug/nJetNNMgamgamWITHDiphotBDT'
#plotDir = '/vols/build/cms/jwd18/BDT/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/debug/recoPlotsWithFits'

f_output = r.TFile.Open("DEBUGhist.root","RECREATE") 

class Bests:
  '''Class to store and update best values during a category optimisation'''

  def __init__(self, nCats):
    self.nCats = nCats

    self.totSignif = -999.
    self.sigs      = [-999. for i in range(self.nCats)]
    self.bkgs      = [-999. for i in range(self.nCats)]
    self.nons      = [-999. for i in range(self.nCats)]
    self.signifs   = [-999. for i in range(self.nCats)]

  def update(self, sigs, bkgs, nons):
    signifs = []
    totSignifSq = 0.
    # Calculate sig for each category
    # From this,compute total sig squared i.e. sig squared for each category, summed up
    # Then square root at the end to get final total Sig for that single BIN
    for i in range(self.nCats):
      sig = sigs[i] 
      bkg = bkgs[i] 
      non = nons[i] 
      signif = self.getAMS(sig, bkg+non)
      #signif = self.getAMS(sig, bkg+(2.*non)) #FIXME trying to give stronger penalty to ggH term
      #signif = self.getAMS(sig, bkg+(non*non)) #FIXME trying to give stronger penalty to ggH term
      signifs.append(signif)
      totSignifSq += signif*signif
    totSignif = np.sqrt( totSignifSq )
    # If this TotalSignif is greater than previous Totalsignif, update Totalsignif, and the signal,bg cat counts
    if totSignif > self.totSignif:
      self.totSignif = totSignif
      for i in range(self.nCats):
        self.sigs[i]     = sigs[i]
        self.bkgs[i]     = bkgs[i]
        self.nons[i]     = nons[i]
        self.signifs[i]  = signifs[i]
      return True
    else:
      return False

  def getAMS(self, s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
      val = (s + b)*np.log(1. + (s/b))
      val = 2*(val - s)
      val = np.sqrt(val)
    return val

  def getSigs(self):
    return self.sigs

  def getBkgs(self):
    return self.bkgs

  def getSignifs(self):
    return self.signifs

  def getTotSignif(self):
    return self.totSignif


class CatOptim:
  '''
  Class to run category optimisation via random search for arbitrary numbers of categories and input discriminator distributions
                            _
                           | \
                           | |
                           | |
      |\                   | |
     /, ~\                / /
    X     `-.....-------./ /
     ~-. ~  ~              |
        \     Optim   /    |
         \  /_     ___\   /
         | /\ ~~~~~   \ |
         | | \        || |
         | |\ \       || )
        (_/ (_/      ((_/
  '''

  def __init__(self, sigWeights, sigMass, sigDiscrims, bkgWeights, bkgMass, bkgDiscrims, nCats, ranges, names):
    '''Initialise with the signal and background weights (as np arrays), then three lists: the discriminator arrays, the ranges (in the form [low, high]) and the names'''
    self.sigWeights    = sigWeights
    self.sigMass       = sigMass
    self.bkgWeights    = bkgWeights
    self.bkgMass       = bkgMass
    self.nonSigWeights = None
    self.nonSigMass    = None
    self.cat6AMS       = {}
    self.nCats         = int(nCats)
    self.bests         = Bests(self.nCats)
    self.sortOthers    = False
    self.addNonSig     = False
    self.transform     = False
    assert len(bkgDiscrims) == len(sigDiscrims)
    assert len(ranges)      == len(sigDiscrims)
    assert len(names)       == len(sigDiscrims)
    self.names          = names
    self.sigDiscrims    = od()
    self.bkgDiscrims    = od()
    self.nonSigDiscrims = None
    self.lows           = od()
    self.highs          = od()
    self.boundaries     = od()
    # populate dictionary with (key:name, value=number)
    #initialise dicts
    for iName,name in enumerate(self.names):
      self.sigDiscrims[ name ] = sigDiscrims[iName] #NB RHS is a value e.g. sigDiscrims[0] = 0.5
      self.bkgDiscrims[ name ] = bkgDiscrims[iName]
      assert len(ranges[iName]) == 2
      #ranges is a list of lists i.e. [ [low,high], [low2,high2],... ]
      self.lows[ name ]       = ranges[iName][0]
      self.highs[ name ]      = ranges[iName][1]
      self.boundaries[ name ] = [-999. for i in range(self.nCats)]

  def setNonSig(self, nonSigWeights, nonSigMass, nonSigDiscrims):
    self.addNonSig      = True
    self.nonSigWeights  = nonSigWeights
    self.nonSigMass     = nonSigMass
    self.nonSigDiscrims = od()
    for iName,name in enumerate(self.names):
      self.nonSigDiscrims[ name ] = nonSigDiscrims[iName]

  def setTransform( self, val ):
    self.transform = val

  def doTransform( self, arr ):
    arr = 1. / ( 1. + np.exp( 0.5*np.log( 2./(arr+1.) - 1. ) ) )
    return arr

  def optimise(self, lumi, nIters, classNo):
    '''Run the optimisation for a given number of iterations'''
    for iIter in range(nIters):
      cuts = od()
      # chose random values between the low and high ranges given, for each name.
      # Usually name=diphoBDT (score) only.
      # Syntax is: .unform(lowestNo, HighestNo, HowManyNumers)
      for iName,name in enumerate(self.names):
        tempCuts = np.random.uniform(self.lows[name], self.highs[name], self.nCats)
        if iName==0 or self.sortOthers:
          tempCuts.sort()
        cuts[name] = tempCuts
        #if self.transform:
        #  tempCuts = self.doTransform(tempCuts)
      sigs = []
      bkgs = []
      nons = []
      if classNo==5: #do class 5 manually (print out some values)
        diphoCuts1 = np.linspace(0.5,1,101) #higher cut. nominally: (0.5,1,101)
        diphoCuts2 = np.linspace(0.5,1,101) #lower cut. nominally: (0.5,1,101)
        sigWeights = self.sigWeights
        bkgWeights = self.bkgWeights
        for diphoCut1 in diphoCuts1: #upper category
          sigWeightTemp1 = sigWeights * (self.sigDiscrims[name]>diphoCut1) 
          bkgWeightTemp1 = bkgWeights * (self.bkgDiscrims[name]>diphoCut1) 
          for diphoCut2 in diphoCuts2: #lower category
            sigWeightTemp2 = sigWeights * (self.sigDiscrims[name]<diphoCut1) * (self.sigDiscrims[name]>diphoCut2)
            bkgWeightTemp2 = bkgWeights * (self.bkgDiscrims[name]<diphoCut1) * (self.bkgDiscrims[name]>diphoCut2)
            #calculate efficiencies for each of the two signal and BG sets then combine in quadrature
            #cut 1 and upward
            if diphoCut1>diphoCut2 : #makes no sense if diphoCut2>diphoCut1 
              sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
              fill_hist(sigHist, self.sigMass, weights=sigWeightTemp1)
              sigCount = 0.68 * lumi * sigHist.Integral() 
              sigDict = self.getRealSigma(sigHist) # return sigma and histo with fit
              sigWidth = sigDict['sigma']
              bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
              fill_hist(bkgHist, self.bkgMass, weights=bkgWeightTemp1)
              bkgDict = self.computeBkg(bkgHist, sigWidth) #return bkg histogram with fit
              bkgCount = bkgDict['bkgCounts']
              AMSupper = self.getAMS(sigCount, bkgCount)
              #print('cut 1: %.4f' % diphoCut1)
              #print('sig count %.4f' % sigCount)
              #print('bkg count %.4f' % bkgCount)
              #print('AMS upper: %.4f' % AMSupper)

              #higher than cut 2, lower than cut 1
              sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
              fill_hist(sigHist, self.sigMass, weights=sigWeightTemp2)
              sigCount = 0.68 * lumi * sigHist.Integral() 
              sigDict = self.getRealSigma(sigHist) # return sigma and histo with fit
              sigWidth = sigDict['sigma']
              bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
              fill_hist(bkgHist, self.bkgMass, weights=bkgWeightTemp2)
              bkgDict = self.computeBkg(bkgHist, sigWidth) #return bkg histogram with fit
              bkgCount = bkgDict['bkgCounts']
              AMSlower = self.getAMS(sigCount, bkgCount)
              #print('cut 2: %.4f' % diphoCut2)
              #print('sig count %.4f' % sigCount)
              #print('bkg count %.4f' % bkgCount)
              #print('AMS lower: %.4f' % AMSlower)
              
              #fill dict iwth {(cut1, cut2) = AMS}
              AMStot = (((AMSupper)**2) + ((AMSlower)**2))**(1./2.)
              self.cat6AMS[(diphoCut1, diphoCut2)] = AMStot
              #print('total AMS %.3f' % AMStot)
      else:
       for iCat in range(self.nCats): #number of sub-cats we are splitting the cat into
         lastCat = (iCat+1 == self.nCats) #boolean, set to true if on last cat
         sigWeights = self.sigWeights
         bkgWeights = self.bkgWeights
         if self.addNonSig: nonSigWeights = self.nonSigWeights
         for iName,name in enumerate(self.names):
           sigWeights = sigWeights * (self.sigDiscrims[name]>cuts[name][iCat])
           bkgWeights = bkgWeights * (self.bkgDiscrims[name]>cuts[name][iCat])
           if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[name]>cuts[name][iCat])
           if not lastCat:
             if iName==0 or self.sortOthers:
               sigWeights = sigWeights * (self.sigDiscrims[name]<cuts[name][iCat+1])
               bkgWeights = bkgWeights * (self.bkgDiscrims[name]<cuts[name][iCat+1])
               if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[name]<cuts[name][iCat+1])
         sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
         fill_hist(sigHist, self.sigMass, weights=sigWeights)
         sigCount = 0.68 * lumi * sigHist.Integral() 
         sigDict = self.getRealSigma(sigHist) # return sigma and histo with fit
         sigWidth = sigDict['sigma']
         sigHist  = sigDict['sigHistFit'] #update signal histogram to include fit
         bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
         fill_hist(bkgHist, self.bkgMass, weights=bkgWeights)
         bkgDict = self.computeBkg(bkgHist, sigWidth) #return bkg histogram with fit
         bkgCount = bkgDict['bkgCounts']
         bkgHist  = bkgDict['bkgHistoFit'] #update bkg hist to include fit
         if self.addNonSig:
           nonSigHist = r.TH1F('nonSigHistTemp','nonSigHistTemp',160,100,180)
           fill_hist(nonSigHist, self.nonSigMass, weights=nonSigWeights)
           nonSigCount = 0.68 * lumi * nonSigHist.Integral() 
         else:
           nonSigCount = 0.
         sigs.append(sigCount)
         bkgs.append(bkgCount)
         nons.append(nonSigCount)
       if self.bests.update(sigs, bkgs, nons):
         #DEBUG: print best fits
         #r.gStyle.SetOptStat(1111)
         #canv = r.TCanvas()
         #sigHist.Draw()
         #canv.Print('%s/signalHistCat%i.pdf' % (plotDir,classNo)) #will overwrite until best is stored
         #bkgHist.Draw()
         #canv.Print('%s/backgroundHistCat%i.pdf' % (plotDir,classNo)) #will overwrite until best is stored
         #with open('%s/fitInfoCat%i.txt' %(plotDir,classNo),'w+') as f: 
         #  f.write('Info: \n sigWidth: %.2f \n signal: %.6f \n background: %.6f'%(sigWidth,sigCount,bkgCount))  
         #end of debug
         for name in self.names:
           self.boundaries[name] = cuts[name]

  def crossCheck(self, lumi, plotDir):
    '''Run a check to ensure the random search found a good mimimum'''
    for iName,name in enumerate(self.names):
      for iCat in range(self.nCats):
        best = self.boundaries[name][iCat]
        rnge = 0.2 * self.highs[name] - self.lows[name]
        graph = r.TGraph()
        for iVal,val in enumerate(np.arange(best-rnge/2., best+rnge/2., rnge/10.)):
          sigs = []
          bkgs = []
          nons = []
          cuts = {} 
          cuts[name] = self.boundaries[name]
          cuts[name][iCat] = val
          bests = Bests(self.nCats)
          for jCat in range(self.nCats):
            lastCat = (jCat+1 == self.nCats)
            sigWeights = self.sigWeights
            bkgWeights = self.bkgWeights
            if self.addNonSig: nonSigWeights = self.nonSigWeights
            for jName,jname in enumerate(self.names):
              sigWeights = sigWeights * (self.sigDiscrims[jname]>cuts[jname][jCat])
              bkgWeights = bkgWeights * (self.bkgDiscrims[jname]>cuts[jname][jCat])
              if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[jname]>cuts[jname][jCat])
              if not lastCat:
                if jName==0 or self.sortOthers:
                  sigWeights = sigWeights * (self.sigDiscrims[jname]<cuts[jname][jCat+1])
                  bkgWeights = bkgWeights * (self.bkgDiscrims[jname]<cuts[jname][jCat+1])
                  if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[jname]<cuts[jname][jCat+1])
            sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
            fill_hist(sigHist, self.sigMass, weights=sigWeights)
            sigCount = 0.68 * lumi * sigHist.Integral() 
            sigWidth = self.getRealSigma(sigHist) 
            bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
            fill_hist(bkgHist, self.bkgMass, weights=bkgWeights)
            bkgCount = self.computeBkg(bkgHist, sigWidth) #fits exp to BG integrates between 125+/-sigma 
            if self.addNonSig:
              nonSigHist = r.TH1F('nonSigHistTemp','nonSigHistTemp',160,100,180)
              fill_hist(nonSigHist, self.nonSigMass, weights=nonSigWeights)
              nonSigCount = 0.68 * lumi * nonSigHist.Integral() 
            else:
              nonSigCount = 0.
            sigs.append(sigCount)
            bkgs.append(bkgCount)
            nons.append(nonSigCount)
          bests.update(sigs, bkgs, nons)
          graph.SetPoint(iVal, val-best, bests.getTotSignif())
        canv = useSty.setCanvas()
        graphName = 'CrossCheck_%s_Cat%g'%(name, iCat)
        graph.SetTitle(graphName.replace('_',' '))
        graph.GetXaxis().SetTitle('Cut value - chosen value')
        graph.GetYaxis().SetTitle('Significance (#sigma)')
        graph.Draw()
        useSty.drawCMS(text='Internal')
        useSty.drawEnPu(lumi=lumi)
        canv.Print('%s/%s.pdf'%(plotDir,graphName))
        canv.Print('%s/%s.png'%(plotDir,graphName))

  def setSortOthers(self, val):
    self.sortOthers = val

  def getBests(self):
    return self.bests
 
  def getPrintableResult(self, iClass):
    printStr = ''
    catNum = self.nCats
    if iClass == 5:
      printStr += str(self.cat6AMS)
      return self.cat6AMS
    else:
      for iCat in reversed(range(self.nCats)):
        catNum = self.nCats - (iCat+1)
        printStr += 'Category %g optimal cuts are:  '%catNum
        for name in self.names:
          printStr += '%s %1.3f,  '%(name, self.boundaries[name][iCat])
        printStr = printStr[:-3]
        printStr += '\n'
        printStr += 'With  S %1.3f,  B %1.3f + %1.3f,  signif = %1.3f \n'%(self.bests.sigs[iCat], self.bests.bkgs[iCat], self.bests.nons[iCat], self.bests.signifs[iCat])
      printStr += 'Corresponding to a total significance of  %1.3f \n\n'%self.bests.totSignif
      return printStr

  def getRealSigma( self, hist ):
    sigma = 2.
    sigFitDict = {}
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
      hist.Fit('gaus')
      fit = hist.GetFunction('gaus')
      sigma = fit.GetParameter(2)
    sigFitDict['sigma'] = sigma 
    sigFitDict['sigHistFit'] = hist #note if no fit done, just return old hist
    return sigFitDict
  
  def computeBkg( self, hist, effSigma ):
    bkgVal = 9999.
    bkgFitDict = {}
    #if hist.GetEntries() > 10 and hist.Integral()>0.000001: #previous
    if hist.GetEffectiveEntries() > 10 and hist.Integral()>0.000001:
      hist.Fit('expo')
      fit = hist.GetFunction('expo')
      bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    bkgFitDict['bkgCounts'] = bkgVal
    bkgFitDict['bkgHistoFit'] = hist
    return bkgFitDict

  def getAMS(self, s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
      val = (s + b)*np.log(1. + (s/b))
      val = 2*(val - s)
      val = np.sqrt(val)
    return val

