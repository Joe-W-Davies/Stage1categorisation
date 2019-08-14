from math import log

def addPt(row):
    return row['CMS_hgg_mass']*row['diphoptom']

def truthDipho(row):
    if not row['stage1cat']==0: return 1
    else: return 0

def truthClass(row): 
    # Previous Stage 1 definitions
    #if not row['stage1cat']==0: return int(row['stage1cat']-3) 
    #else: return 0

    #Stage1.1 definitions
    # NB dont need a > 0 cut on things like Hjj because if we have two jets, this is non zero
  if(row['gen_pTH'] < 200):
    if(row['n_gen_jets'] == 0):
      if(row['gen_pTH'] < 10): return 0
      else: return 1
    if(row['n_gen_jets'] == 1): 
      if(row['gen_pTH'] < 60): return 2
      elif(row['gen_pTH'] < 120): return 3
      elif(row['gen_pTH'] < 200): return 4 
    if(row['n_gen_jets'] >= 2): 
      if(row['gen_dijet_Mjj'] < 350):
        if(row['gen_pTH'] < 60): return 5 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
        elif(row['gen_pTH'] < 120): return 6
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
        elif(row['gen_pTH'] < 200): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
      else: #( implicit if Mjj>350)
        if(row['gen_ptHjj'] < 25): 
          if(row['gen_dijet_Mjj'] < 700): return 9
          else: return 10
        else:
          if(row['gen_dijet_Mjj'] < 700): return 11
          else: return 12
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
  elif(row['gen_pTH']>200): return 8
  else: return -1 #everything that doesn't go into a bin

def truthJets(row):
    # previous definitions
    #if row['stage1cat']==3: return 0
    #elif row['stage1cat']>=4 and row['stage1cat']<=7: return 1
    #elif row['stage1cat']>=8 and row['stage1cat']<=11: return 2
    
    if row['n_gen_jets']==0: return 0
    if row['n_gen_jets']==1: return 1
    if row['n_gen_jets']>=2: return 2

    else: return -1

def jetPtToClass(row):
    #Stage1.1 reco/cat definitions
  if(row['diphopt'] < 200):
    if(row['n_pred_jets'] == 0):
      if(row['diphopt'] < 10): return 0
      else: return 1
    if(row['n_pred_jets'] == 1): 
      if(row['diphopt'] < 60): return 2
      elif(row['diphopt'] < 120): return 3
      elif(row['diphopt'] < 200): return 4 
    if(row['n_pred_jets'] >= 2): 
      if(row['dijet_Mjj'] < 350):
        if(row['diphopt'] < 60): return 5 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
        elif(row['diphopt'] < 120): return 6
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
        elif(row['diphopt'] < 200): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
      else: #( implicit if Mjj>350)
        if(row['ptHjj'] < 25):
          if(row['dijet_Mjj'] < 700): return 9
          else: return 10
        else: #(implicit if PtHjj > 25)
          if(row['dijet_Mjj'] < 700): return 11
          else: return 12
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
  elif(row['diphopt']>200): return 8
  else: return -1 #everything that doesn't go into a bin
  
def jetPtToggHClass(row):
    #Stage1.1 reco/cat definitions
  if(row['diphopt'] < 200):
    if(row['n_pred_jets'] == 0):
      if(row['diphopt'] < 10): return 0
      else: return 1
    if(row['n_pred_jets'] == 1): 
      if(row['diphopt'] < 60): return 2
      elif(row['diphopt'] < 120): return 3
      elif(row['diphopt'] < 200): return 4 
    if(row['n_pred_jets'] >= 2): 
      if(row['diphopt'] < 60): return 5 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
      elif(row['diphopt'] < 120): return 6
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
      elif(row['diphopt'] < 200): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
  elif(row['diphopt']>200): return 8
  else: return -1 #everything that doesn't go into a bin

def reco(row): 
   #if row['n_rec_jets']==0: return 0
   #elif row['n_rec_jets']==1:
   #    if row['diphopt'] < 60: return 1
   #    elif row['diphopt'] < 120: return 2
   #    elif row['diphopt'] < 200: return 3
   #    else: return 4
   #else: # i.e. for jets and above
   #    if row['diphopt'] < 60: return 5
   #    elif row['diphopt'] < 120: return 6 #elif means below 120 but not below 60, so gets correct region
   #    elif row['diphopt'] < 200: return 7
   #    else: return 8

    #Stage1.1 reco/cat definitions
  if(row['diphopt'] < 200):
    if(row['n_rec_jets'] == 0):
      if(row['diphopt'] < 10): return 0
      else: return 1
    if(row['n_rec_jets'] == 1): 
      if(row['diphopt'] < 60): return 2
      elif(row['diphopt'] < 120): return 3
      elif(row['diphopt'] < 200): return 4 
    if(row['n_rec_jets'] >= 2): 
      if(row['dijet_Mjj'] < 350):
        if(row['diphopt'] < 60): return 5 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
        elif(row['diphopt'] < 120): return 6
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
        elif(row['diphopt'] < 200): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
      else: #( implicit if Mjj>350)
        if(row['ptHjj'] < 25):
          if(row['dijet_Mjj'] < 700): return 9
          else: return 10
        else: #(implicit if PtHjj > 25)
          if(row['dijet_Mjj'] < 700): return 11
          else: return 12
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
  elif(row['diphopt']>200): return 8
  else: return -1 #everything that doesn't go into a bin

def diphoWeight(row, sigWeight=1.):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    elif row['stage1cat'] > 0.01:
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
    weightFactors = [0.0002994, 0.0000757, 0.0000530, 0.0000099, 0.0000029, 0.0000154, 0.0000235, 0.0000165, 0.0000104] #FIXME update these
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

def jetWeight(row): #equalise the weight for each jet type i.e. equal sum of weights for the 3 classes
    #weightFactors = [0.606560, 0.270464, 0.122976] 
    weightFactors = [0.609415, 0.271737, 0.118849] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs. 
    #weight = abs(weight)
    return weight

def sqrtJetWeight(row): #variation on equal weight
    weightFactors = [0.780650, 0.521284, 0.344745] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs. 
    #weight = abs(weight)
    return weight

def cbrtJetWeight(row): #variation on equal weight
    weightFactors = [0.847821, 0.647713, 0.491660] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs. 
    #weight = abs(weight)
    return weight

def procWeight(row): #equalise the weights for the procs i.e. equal sum of weights for the 9 classes
    weightFactors = [0.134534, 0.488618, 0.147402, 0.095248, 0.016962, 0.028589, 0.040634, 0.027472, 0.020542] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthClass']) ] 
    #weight = abs(weight)
    return weight

def sqrtProcWeight(row): #equalise the weights for the procs i.e. equal sum of weights for the 9 classes
    weightFactors = [0.366789, 0.699012, 0.383930, 0.308623, 0.130238, 0.169083, 0.201579, 0.165748, 0.143325] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthClass']) ] 
    #weight = abs(weight)
    return weight

def cbrtProcWeight(row): #equalise the weights for the procs i.e. equal sum of weights for the 9 classes
    weightFactors = [0.512402, 0.787632, 0.528244, 0.456687, 0.256936, 0.305773, 0.343793, 0.301738, 0.143325] 
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthClass']) ] 
    #weight = abs(weight)
    return weight

#def altDiphoWeight(row, sigWeight=1./0.001169):
def altDiphoWeight(row, sigWeight=1./0.001297):
    weight = row['weight']
    if row['proc'].count('qcd'):
        weight *= 0.04 #downweight bc too few events
    elif row['stage1cat'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight
