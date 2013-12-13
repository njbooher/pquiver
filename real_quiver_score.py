from ConsensusCore import *
import numpy as np
from projutils import getReads, sanity_check
from pbcore.io import CmpH5Reader
from GenomicConsensus import reference

cmpH5 = CmpH5Reader('/home/nick/workspace/btry6790_project/PXO99A_ref_wo_one_copy_212kb_repeat.cmp.h5')
reference.loadFromFile("/home/nick/workspace/btry6790_project/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat/sequence/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat.fasta", cmpH5)

tmplSeq, realTmplLen, fwdSeqs, qvInfo = getReads(cmpH5, reference, (146000, 146050), 64, 100, real_quiver=True)
np.set_printoptions(linewidth=200)

totalScore = 0
for read in range(len(fwdSeqs)):
    
    features = QvSequenceFeatures(fwdSeqs[read],
                                   FloatFeature(qvInfo[0, :len(fwdSeqs[read]), read].astype(np.float32)),
                                   FloatFeature(qvInfo[4, :len(fwdSeqs[read]), read].astype(np.float32)),
                                   FloatFeature(qvInfo[2, :len(fwdSeqs[read]), read].astype(np.float32)),
                                   FloatFeature(qvInfo[3, :len(fwdSeqs[read]), read].astype(np.float32)),
                                   FloatFeature(qvInfo[1, :len(fwdSeqs[read]), read].astype(np.float32)))
    
    params = QvModelParams(0.2627555,
                           -1.09688872,
                           -0.01637988,
                           -0.60275947,
                           -0.02682689,
                           -1.00012494,
                           0.06000148,
                           -0.02579358,
                           -0.15864559,
                           -0.04403654,
                           -1.02398814,
                           -0.12135255)
    
    I = features.Length()
    J = len(tmplSeq)
    a = DenseMatrix(I+1, J+1)
    b = DenseMatrix(I+1, J+1)
    
    e = QvEvaluator(features, tmplSeq, params)
    
    r = SimpleQvRecursor(ALL_MOVES, BandingOptions(0, 0))
    r.FillAlphaBeta(e, a, b)
    
    print("Read %d: %s" % (read, str(a.Get(I,J))))
    totalScore += a.Get(I,J)

print(totalScore)