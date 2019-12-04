def addPt(row):
    return row['CMS_hgg_mass']*row['diphoptom']

def truthDipho(row):
    if not row['HTXSstage1_1_cat']==0: return 1
    else: return 0

def truthVhHad(row):
    if row['HTXSstage1_1_cat']==204: return 1
    else: return 0
    #elif row['HTXSstage1_1_cat']>107 and row['HTXSstage1_1_cat']<111: return 0
    #else: return -1

def vhHadWeight(row, ratio):
    weight = 1000. * abs(row['weight'])
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVhHad']==1: 
      return ratio * weight
    else: return weight

def truthVBF(row):
    if row['HTXSstage1_1_cat']>206.5 and row['HTXSstage1_1_cat']<210.5: return 2
    elif row['HTXSstage1_1_cat']>109.5 and row['HTXSstage1_1_cat']<113.5: return 1
    elif row['HTXSstage1_1_cat']==0: return 0
    else: return -1

def vbfWeight(row, vbfSumW, gghSumW, bkgSumW):
    weight = abs(row['weight'])
    if row['proc'].count('qcd'): 
        weight *= 0.04 
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVBF']==2: 
      return (bkgSumW/vbfSumW) * weight
    elif row['truthVBF']==1: 
      return (bkgSumW/gghSumW) * weight
    else: return weight

def truthClass(row):
    if not row['HTXSstage1_1_cat']==0: return int(row['HTXSstage1_1_cat']-3)
    else: return 0

def truthJets(row):
    if row['HTXSstage1_1_cat']==3: return 0
    elif row['HTXSstage1_1_cat']>=4 and row['HTXSstage1_1_cat']<=7: return 1
    elif row['HTXSstage1_1_cat']>=8 and row['HTXSstage1_1_cat']<=11: return 2
    else: return -1

def reco(row):
    if row['n_rec_jets']==0: return 0
    elif row['n_rec_jets']==1:
        if row['diphopt'] < 60: return 1
        elif row['diphopt'] < 120: return 2
        elif row['diphopt'] < 200: return 3
        else: return 4
    else:
        if row['diphopt'] < 60: return 5
        elif row['diphopt'] < 120: return 6
        elif row['diphopt'] < 200: return 7
        else: return 8

def diphoWeight(row, sigWeight=1.):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1_1_cat'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def combinedWeight(row):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    weight = abs(weight)
    return weight

def normWeight(row, bkgWeight=100., zerojWeight=1.):
    weightFactors = [0.0002994, 0.0000757, 0.0000530, 0.0000099, 0.0000029, 0.0000154, 0.0000235, 0.0000165, 0.0000104] #update these at some point
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 / weightFactors[ int(row['truthClass']) ] #reduce because too large by default
    else: 
        weight *= 1. / weightFactors[ int(row['truthClass']) ] #otherwise just reweight by xs
    weight = abs(weight)
    #arbitrary weight changes to be optimised
    if row['proc'] != 'ggh':
        weight *= bkgWeight
    elif row['reco'] == 0: 
        weight *= zerojWeight
    return weight

def jetWeight(row):
    weightFactors = [0.606560, 0.270464, 0.122976]
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs
    weight = abs(weight)
    return weight

def altDiphoWeight(row, weightRatio):
    weight = row['weight']
    if row['proc'].count('qcd'):
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1_1_cat'] > 0.01:
        weight *= weightRatio #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight
