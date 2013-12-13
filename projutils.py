import GenomicConsensus
from GenomicConsensus import utils, reference, windows
from GenomicConsensus.utils import readsInWindow
from GenomicConsensus.windows import subWindow
from GenomicConsensus.quiver.utils import lifted
from GenomicConsensus.quiver.model import AllQVsModel
from pbcore.io import CmpH5Reader
import ConsensusCore as cc
import numpy as np

#https://github.com/PacificBiosciences/GenomicConsensus/blob/master/GenomicConsensus/quiver/

#filterAlns from GenomicConsensus.quiver.utils with readStumpinessThreshold param instead of quiverConfig
def filterAlns(refWindow, alns, readStumpinessThreshold):
    """
    Given alns (already clipped to the window bounds), filter out any
    that are incompatible with Quiver.

    By and large we avoid doing any filtering to avoid potential
    reference bias in variant calling.

    However at the moment the POA (and potentially other components)
    breaks when there is a read of zero length.  So we throw away
    reads that are "stumpy", where the aligner has inserted a large
    gap, such that while the alignment technically spans the window,
    it may not have any read content therein:

          Ref   ATGATCCAGTTACTCCGATAAA
          Read  ATG---------------TA-A
          Win.     [              )
    """
    return [ a for a in alns
             if a.readLength >= (readStumpinessThreshold * a.referenceSpan) ]

def getReads(cmpH5, reference, interval, paddedTemplateWidth, depthLimit, real_quiver=False):
    
    minMapQV = 10
    minPoaCoverage = 3
    maxPoaCoverage = 11
    mutationSeparation = 10
    mutationNeighborhood = 20
    maxIterations = 20
    refineDinucleotideRepeats = True
    noEvidenceConsensus = "nocall"
    computeConfidence = True
    readStumpinessThreshold = 0.1
    
    refId = [x for x in reference.enumerateIds()][0]
    refSeq = reference.byId[refId].sequence
    refWindow = (refId, 0, reference.byId[refId].length)
    
    intStart, intEnd = interval
    subWin = subWindow(refWindow, interval)
    
    windowRefSeq = refSeq[intStart:intEnd]
    rows = readsInWindow(cmpH5, subWin,
                           depthLimit = depthLimit,
                           minMapQV = minMapQV,
                           strategy = "longest",
                           stratum = None,
                           barcode = None)
    
    #print([cmpH5[row].alignedLength for row in rows if cmpH5[row].spansReferenceRange(intStart, intEnd)])
    spanningRows = [row for row in rows if cmpH5[row].spansReferenceRange(intStart, intEnd) ]
    
    alns = cmpH5[spanningRows]
    clippedAlns_ = [ aln.clippedTo(*interval) for aln in alns ]
    clippedAlns__ = [ aln for aln in clippedAlns_ if aln.alignedLength <= paddedTemplateWidth - 7]
    clippedAlns = filterAlns(subWin, clippedAlns__, readStumpinessThreshold)
    
    # Compute the POA consensus, which is our initial guess, and
    # should typically be > 99.5% accurate
    fwdSequences = [ a.read(orientation="genomic", aligned=False)
                     for a in clippedAlns]
    
    p = cc.PoaConsensus.FindConsensus(fwdSequences[:maxPoaCoverage])
    
    template = p.Sequence()
    
    tmplSeq = np.zeros((paddedTemplateWidth), dtype=np.uint8)
    tmplOrds = map(ord, template)
    tmplSeq[:len(tmplOrds)] = tmplOrds
    
    #read pos y, read x
    readSeqs = np.zeros((paddedTemplateWidth, len(clippedAlns)), dtype=np.uint8)
    
    for i in xrange(len(clippedAlns)):
        alnOrds = map(ord, fwdSequences[i])
        readSeqs[:len(alnOrds), i] = alnOrds
    
    #uint8
    #metric z, read pos y, read x
    qvInfo = np.zeros((8, paddedTemplateWidth, len(clippedAlns)), dtype=np.uint8)
    
    for i in xrange(len(clippedAlns)):
        qvInfo[0, :clippedAlns[i].readLength, i] = clippedAlns[i].InsertionQV(orientation="genomic", aligned=False)
        qvInfo[1, :clippedAlns[i].readLength, i] = clippedAlns[i].MergeQV(orientation="genomic", aligned=False)
        qvInfo[2, :clippedAlns[i].readLength, i] = clippedAlns[i].DeletionQV(orientation="genomic", aligned=False)
        qvInfo[3, :clippedAlns[i].readLength, i] = clippedAlns[i].DeletionTag(orientation="genomic", aligned=False)
        qvInfo[4, :clippedAlns[i].readLength, i] = clippedAlns[i].SubstitutionQV(orientation="genomic", aligned=False)
    
    if real_quiver:
        return template, len(tmplOrds), fwdSequences, qvInfo
    else:
        return tmplSeq, len(tmplOrds), readSeqs, qvInfo

def sanity_check(tmplSeq, realTmplLen, readSeqs, qvInfo):
    
    flatReadSeqs = readSeqs.flatten()
    
    numReadPos, numReads = readSeqs.shape
    
    for i in xrange(numReadPos):
        pos = i * numReads
        
        if not (np.array_equal(flatReadSeqs[pos:pos+numReads], readSeqs[i, :])):
            print("ERROR")
    
    flatQVInfo = qvInfo.flatten()
    
    numMetrics, numReadPos, numReads = qvInfo.shape
    
    for i in xrange(numMetrics):
        for j in xrange(numReadPos):
            pos = i * numReadPos * numReads + j * numReads
            if not (np.array_equal(flatQVInfo[pos:pos+numReads], qvInfo[i, j, :])):
                print("ERROR")
    
    #flatten()