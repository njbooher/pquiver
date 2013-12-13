import numpy as np
from pbcore.io import CmpH5Reader
from GenomicConsensus import reference
from projutils import getReads
from bqcy.bqcy import getTemplateScore
from bqfast.bqfast import run_bqfast

cmpH5 = CmpH5Reader('/home/nick/workspace/btry6790_project/PXO99A_ref_wo_one_copy_212kb_repeat.cmp.h5')
reference.loadFromFile("/home/nick/workspace/btry6790_project/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat/sequence/ref_PXO99A_genome_reference_wo_one_copy_212k_repeat.fasta", cmpH5)

#tmplSeq, realTmplLen, readSeqs, qvInfo = getReads(cmpH5, reference, (146000, 146100), 128, 100)
tmplSeq, realTmplLen, readSeqs, qvInfo = getReads(cmpH5, reference, (146000, 146050), 64, 100)

print("Real Template: " + ''.join(map(chr, tmplSeq.tolist())))

tmplSeq = np.zeros((64), dtype=np.uint8)
tmplOrds = map(ord, "A" * 50)
tmplSeq[:len(tmplOrds)] = tmplOrds

tmplScore = getTemplateScore(tmplSeq, readSeqs, qvInfo)
results, polishedTmplSeq = run_bqfast(tmplSeq, tmplSeq.shape[0], tmplScore, readSeqs, qvInfo)

print("Polished: " + ''.join(map(chr, polishedTmplSeq.tolist())))
print("Fake Template: " + ''.join(map(chr, tmplSeq.tolist())))


print(results[:tmplSeq.shape[0] * 8])