import numpy as np
from projutils import getReads

tmplSeq, realTmplLen, readSeqs, qvInfo = getReads()
#sanity_check(tmplSeq, realTmplLen, readSeqs, qvInfo)


#[C2.AllQVsModel]
# s stands for slope
Match            =  0.2627555
Mismatch         = -1.09688872
MismatchS        = -0.01637988
Branch           = -0.60275947
BranchS          = -0.02682689
DeletionN        = -1.00012494
DeletionWithTag  =  0.06000148
DeletionWithTagS = -0.02579358
Nce              = -0.15864559
NceS             = -0.04403654
Merge            = -1.02398814
MergeS           = -0.12135255

#metrics
InsertionIdx = 0
MergeIdx = 1
DeletionIdx = 2
DeletionTagIdx = 3
SubstitutionIdx = 4

readLength = 256
tmplLength = 256

#read = 0

for read in range(readSeqs.shape[1]):
    
    alpha = np.zeros((readLength + 1, tmplLength + 1), dtype=np.float64)
    
    score = 0.0
    
    for i in range(readLength + 1):
        if i > 0 and readSeqs[i - 1, read] == 0:
            lastAlphaI = i - 1
            break
        for j in range(tmplLength + 1):
            if j > 0 and tmplSeq[j - 1] == 0:
                lastAlphaJ = j - 1
                break
            
            moveScore = 0
            
            #Incorporate
            if i > 0 and j > 0:
                if readSeqs[i - 1, read] == tmplSeq[j - 1]:
                    moveScore = alpha[i - 1, j - 1] + Match
                else:
                    moveScore = alpha[i - 1, j - 1] + Mismatch + MismatchS * qvInfo[SubstitutionIdx, i - 1, read]
                score = max(score, moveScore)
            
            #Delete
            if j > 0:
                if i < readLength and qvInfo[DeletionTagIdx, i, read] == tmplSeq[j - 1]:
                    moveScore = alpha[i, j - 1] + DeletionWithTag + DeletionWithTagS * qvInfo[DeletionIdx, i, read]
                else:
                    moveScore = alpha[i, j - 1] + DeletionN
                score = max(score, moveScore)
            
            #Extra
            
            if i > 0:
                if j < tmplLength and readSeqs[i - 1, read] == tmplSeq[j]:
                    moveScore = alpha[i - 1, j] + Branch + BranchS * qvInfo[InsertionIdx, i - 1, read]
                else:
                    moveScore = alpha[i - 1, j] + Nce + NceS * qvInfo[InsertionIdx, i - 1, read]
                score = max(score, moveScore)
            
            #Merge
            if i > 0 and j > 1:
                if not (readSeqs[i - 1, read] == tmplSeq[j - 2] and readSeqs[i - 1, read] == tmplSeq[j - 1]):
                    moveScore = alpha[i - 1, j - 2] + float("-inf")
                else:
                    moveScore = alpha[i - 1, j - 2] + Merge + MergeS * qvInfo[MergeIdx, i - 1, read]
                score = max(score, moveScore)
            
            alpha[i, j] = score
            
    print("Read %d: %s" % (read, str(alpha[lastAlphaI, lastAlphaJ])))