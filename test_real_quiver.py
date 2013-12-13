import GenomicConsensus
from GenomicConsensus import utils, reference, windows
from GenomicConsensus.utils import readsInWindow
from GenomicConsensus.windows import subWindow
from GenomicConsensus.quiver.utils import filterAlns, consensusForAlignmentsDisregardPOA
from GenomicConsensus.quiver.quiver import configure
from pbcore.io import CmpH5Reader
import ConsensusCore as cc
import numpy as np

class dummy(object):
    pass

options = dummy()
options.diploid = False
options.parametersFile = "/home/nick/workspace/btry6790_project/venv/lib/python2.7/site-packages/GenomicConsensus/quiver/resources/2013-09/GenomicConsensus/QuiverParameters.ini"
options.parameterSet = "best"
options.refineDinucleotideRepeats = True
options.noEvidenceConsensusCall = "nocall"
options.minMapQV = 10
options.fastMode = False

cmpH5 = CmpH5Reader('/home/nick/workspace/btry6790_project/PXO99A_ref_wo_one_copy_212kb_repeat.cmp.h5')
quiverConfig = configure(options, cmpH5)

depthLimit = 100

reference.loadFromFile("/home/nick/workspace/btry6790_project/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat/sequence/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat.fasta", cmpH5)
refId = [x for x in reference.enumerateIds()][0]
refSeq = reference.byId[refId].sequence
refWindow = (refId, 0, reference.byId[refId].length)

def run_real_quiver(cmpH5, quiverConfig, interval, depthLimit, refSeq, refWindow, seedConsensus):
    
    intStart, intEnd = interval
    subWin = subWindow(refWindow, interval)
    
    windowRefSeq = refSeq[intStart:intEnd]
    rows = readsInWindow(cmpH5, subWin,
                           depthLimit = depthLimit,
                           minMapQV = quiverConfig.minMapQV,
                           strategy = "longest",
                           stratum = None,
                           barcode = None)
    
    spanningRows = [row for row in rows if cmpH5[row].spansReferenceRange(intStart, intEnd) ]
    
    alns = cmpH5[spanningRows]
    clippedAlns_ = [ aln.clippedTo(*interval) for aln in alns ]
    clippedAlns__ = [ aln for aln in clippedAlns_ if aln.alignedLength <= 120]
    clippedAlns = filterAlns(subWin, clippedAlns__, quiverConfig)
    
    consensus = consensusForAlignmentsDisregardPOA(subWin, windowRefSeq, clippedAlns, quiverConfig, "A"*100)
    print(str(consensus.sequence))
 
run_real_quiver(cmpH5, quiverConfig, (146000, 146100), depthLimit, refSeq, refWindow, "T"*100)