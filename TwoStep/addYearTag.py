import ROOT  as r
from root_numpy import tree2array, array2tree
import numpy as np
from optparse import OptionParser

#configure options
parser = OptionParser()
parser.add_option('-y', '--year', default=None, help='Year of input files')
(opts,args)=parser.parse_args()

#procFileMap = {'ggh':'ggH_powheg.root', 'vbf':'VBF_powheg.root',  
#               'dipho':'Dipho_powheg.root', 'gjet':'GJet_powheg.root', 'qcd':'QCD_powheg.root'}

procFileMap = {'ggh':'ggh_powheg'}
theProcs = procFileMap.keys()

getOppositeYear = {'2016':['2017','2018'], '2017':['2016','2018'], '2018':['2016','2017']}

for proc,fn in procFileMap.iteritems():
   print 'processing %s' %proc
   #trainFile   = r.TFile('/vols/cms/jwd18/Stage1categorisation/Pass1/%s/trees/%s'%(opts.year, fn))
   trainFile   = r.TFile('/vols/cms/jwd18/Stage1categorisation/Pass1/Combined/trees/%s_%s.root'%(fn,opts.year))
   if proc[-1].count('h') or 'vbf' in proc:
     print 'getting tree: vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc
     trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc)
   elif 'dipho' in proc:
     print 'getting tree: vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc
     trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc)
   else: 
     print 'getting tree: vbfTagDumper/trees/%s_anyfake_13TeV_GeneralDipho'%proc
     trainTree = trainFile.Get('vbfTagDumper/trees/%s_anyfake_13TeV_GeneralDipho'%proc)
   #newFile = r.TFile('/vols/cms/jwd18/Stage1categorisation/Pass1/%s/trees/withBinaryYearTag/%s_powheg_withBinaryYear%s.root'%(opts.year, proc, opts.year),'RECREATE')
   newFile = r.TFile('/vols/cms/jwd18/Stage1categorisation/Pass1/Combined/trees/%s_powheg_withBinaryYear_%s.root'%(proc, opts.year),'RECREATE')
   newTree = trainTree.CloneTree()

   nEvents = newTree.GetEntries()
   print nEvents

   #create column to append to tree
   yearTag = np.full( (nEvents,1), 1, dtype=[ ('year_%s'%opts.year, np.int32)] )
   print('%s year tag column:'%opts.year)
   print(yearTag)
   holderOne = array2tree(yearTag, tree=newTree)

   #create additional columns for other two years and set entries to zero
   for oppYear in getOppositeYear[opts.year]:
     oppositeYearTag = np.full( (nEvents,1), 0, dtype=[ ('year_%s'%oppYear, np.int32)] )
     holderTwo = array2tree(oppositeYearTag, tree=newTree)
     print('opposite year tag column for year %s:' %oppYear)
     print(oppositeYearTag)

   newFile.Write()



#add to existing tree using array2tree

