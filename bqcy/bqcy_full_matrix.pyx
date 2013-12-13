cimport cython
import numpy as np
cimport numpy as np
from libc.float cimport FLT_MAX
from libc.stdint cimport uint8_t

ctypedef np.uint8_t[:, :, :] qvInfo_t
ctypedef np.uint8_t[:, :] readSeqs_t
ctypedef np.uint8_t[:] tmplSeq_t
ctypedef np.float32_t[:, :] alpha_t

cdef:
    #[C2.AllQVsModel]
    double Match            =  0.2627555
    double Mismatch         = -1.09688872
    double MismatchS        = -0.01637988
    double Branch           = -0.60275947
    double BranchS          = -0.02682689
    double DeletionN        = -1.00012494
    double DeletionWithTag  =  0.06000148
    double DeletionWithTagS = -0.02579358
    double Nce              = -0.15864559
    double NceS             = -0.04403654
    double Merge            = -1.02398814
    double MergeS           = -0.12135255
    
    #metrics
    int InsertionIdx = 0
    int MergeIdx = 1
    int DeletionIdx = 2
    int DeletionTagIdx = 3
    int SubstitutionIdx = 4
    
    int readLength = 256
    int tmplLength = 256

@cython.boundscheck(False)
def uniqueSingleBaseMutations(tmplSeq_t templateSequence):
    """
    Return an iterator over all single-base mutations of a
    templateSequence that result in unique mutated sequences.
    """
    
    cdef:
        int tplStart
        uint8_t tplBase, subsBase, insBase
        tmplSeq_t mutated = np.zeros((256), dtype=np.uint8)
    
    yield templateSequence
    allBases = map(ord, "ACGT")
    for tplStart in xrange(0, templateSequence.size):
        tplBase     = templateSequence[tplStart]
        prevTplBase = templateSequence[tplStart-1] if (tplStart > 0) else None
        # snvs
        for subsBase in allBases:
            if subsBase != tplBase:
                mutated = templateSequence.copy()
                mutated[tplStart] = subsBase
                yield mutated
        # Insertions---only allowing insertions that are not cognate
        # with the previous base.
        for insBase in allBases:
            if insBase != prevTplBase:
                #mutated = np.append(templateSequence[:tplStart], [insBase])
                mutated[:tplStart] = templateSequence[:tplStart]
                mutated[tplStart] = insBase
                if tplStart < templateSequence.size - 1:
                    mutated[tplStart+1:] = templateSequence[tplStart + 1:]
                    #mutated = np.append(mutated, templateSequence[tplStart + 1:])
                yield mutated
        # Deletion--only allowed if refBase does not match previous tpl base
        if tplBase != prevTplBase:
            mutated[:tplStart] = templateSequence[:tplStart]
            if tplStart < templateSequence.size - 1:
                #mutated = np.append(mutated, templateSequence[tplStart + 1:])
                mutated[tplStart+1:] = templateSequence[tplStart + 1:]
            yield mutated

@cython.boundscheck(False)
def run_bqcy(tmplSeq_t origTmplSeq, readSeqs_t readSeqs, qvInfo_t qvInfo):
    
    cdef:
        int i, j, k, l, read, lastAlphaI, lastAlphaJ
        alpha_t alpha
        double score = 0
        double moveScore = 0
        double totalScore = 0
        double origTmplScore = 0
        double currentTmplScore = 0
        double bestMutantScore = FLT_MAX
        tmplSeq_t bestMutatedSeq = origTmplSeq.copy()
        tmplSeq_t currentTmplSeq 
        tmplSeq_t tmplSeq
    
    for l in range(10):
        
        currentTmplSeq = bestMutatedSeq
        currentTmplScore = bestMutantScore
        bestMutatedSeq = np.zeros((256), dtype=np.uint8)
        bestMutantScore = FLT_MAX
        
        k = 0
        
        for tmplSeq in uniqueSingleBaseMutations(currentTmplSeq):
            
            totalScore = 0
            
            for read in range(readSeqs.shape[1]):
                
                alpha = np.zeros((readLength + 1, tmplLength + 1), dtype=np.float32)
                
                score = (-FLT_MAX)
                
                for i in range(readLength + 1):
                    
                    if i > 0 and readSeqs[i - 1, read] == 0:
                        lastAlphaI = i - 1
                        break
                    
                    for j in range(tmplLength + 1):
                        
                        if j > 0 and tmplSeq[j - 1] == 0:
                            lastAlphaJ = j - 1
                            break
                        
                        score = (-FLT_MAX)
                        
                        if i == 0 and j == 0:
                            score = 0
                        
                        moveScore = -1
                        
                        #Incorporate
                        if i > 0 and j > 0:
                            if readSeqs[i - 1, read] == tmplSeq[j - 1]:
                                moveScore = alpha[i - 1, j - 1] + Match
                            else:
                                moveScore = alpha[i - 1, j - 1] + Mismatch + MismatchS * qvInfo[SubstitutionIdx, i - 1, read]
                            score = max(score, moveScore)
                            
                        #Extra
                        
                        if i > 0:
                            if j < tmplLength and readSeqs[i - 1, read] == tmplSeq[j]:
                                moveScore = alpha[i - 1, j] + Branch + BranchS * qvInfo[InsertionIdx, i - 1, read]
                            else:
                                moveScore = alpha[i - 1, j] + Nce + NceS * qvInfo[InsertionIdx, i - 1, read]
                            score = max(score, moveScore)
                        
                        #Delete
                        if j > 0:
                            if i < readLength and qvInfo[DeletionTagIdx, i, read] == tmplSeq[j - 1]:
                                moveScore = alpha[i, j - 1] + DeletionWithTag + DeletionWithTagS * qvInfo[DeletionIdx, i, read]
                            else:
                                moveScore = alpha[i, j - 1] + DeletionN
                            score = max(score, moveScore)
                        
                        #Merge
                        if i > 0 and j > 1:
                            if not (readSeqs[i - 1, read] == tmplSeq[j - 2] and readSeqs[i - 1, read] == tmplSeq[j - 1]):
                                moveScore = alpha[i - 1, j - 2] + (-FLT_MAX)
                            else:
                                moveScore = alpha[i - 1, j - 2] + Merge + MergeS * qvInfo[MergeIdx, i - 1, read]
                            score = max(score, moveScore)
                        
                        alpha[i, j] = score
                
                if read < 100:
                    totalScore += alpha[lastAlphaI, lastAlphaJ]
                #if read == 1:
                #    print(np.asarray(alpha[lastAlphaI-5:lastAlphaI, lastAlphaJ-5:lastAlphaJ]))
                #print("Read %d: %s" % (read, str(alpha[lastAlphaI, lastAlphaJ])))
            
            if l == 0 and k == 0:
                origTmplScore = totalScore
            
            if k == 0:
                currentTmplScore = totalScore
                bestMutantScore = totalScore
            else:
                if totalScore > bestMutantScore:
                    bestMutatedSeq = tmplSeq
                    bestMutantScore = totalScore
            
            k += 1
    
    return origTmplScore, currentTmplScore,  currentTmplSeq