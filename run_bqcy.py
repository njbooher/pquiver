import numpy as np
from pbcore.io import CmpH5Reader
from GenomicConsensus import reference
from projutils import getReads
from bqcy.bqcy import run_bqcy

cmpH5 = CmpH5Reader('/home/nick/workspace/btry6790_project/PXO99A_ref_wo_one_copy_212kb_repeat.cmp.h5')
reference.loadFromFile("/home/nick/workspace/btry6790_project/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat/sequence/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat.fasta", cmpH5)

tmplSeq, realTmplLen, readSeqs, qvInfo = getReads(cmpH5, reference, (146000, 146050), 64, 100)

#print(readSeqs[:, 65:])
#exit()

print("POA Consensus: " + ''.join(map(chr, tmplSeq.tolist())))

tmplSeq = np.zeros((64), dtype=np.uint8)
tmplOrds = map(ord, "A" * 50)
tmplSeq[:len(tmplOrds)] = tmplOrds

results = np.zeros(8 * tmplSeq.shape[0], dtype=np.float64)
origTmplScore, bestMutantScore,  bestMutatedSeq = run_bqcy(tmplSeq, readSeqs, qvInfo, results)
print("Polished: " + ''.join(map(chr, np.asarray(bestMutatedSeq).tolist())))
print("Fake Template: " + ''.join(map(chr, np.asarray(tmplSeq).tolist())))
print(results)


