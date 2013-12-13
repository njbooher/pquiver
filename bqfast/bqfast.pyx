import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t

ctypedef np.uint8_t[:, :, :] qvInfo_t
ctypedef np.uint8_t[:, :] readSeqs_t
ctypedef np.uint8_t[:] tmplSeq_t

cdef extern from "bqcuda.h":
    int run_bqcuda(uint8_t *origTmplSeq, uint8_t *polishedTmplSeq, uint8_t *readSeqs, uint8_t *qvInfo, double *results, double origTmplScore, int tmplLen, int readLen, int numReads, int numMetrics)

def run_bqfast(tmplSeq_t tmplSeq, int realTmplLen, double tmplScore, readSeqs_t readSeqs, qvInfo_t qvInfo):
    
    cdef int tmplLen, readLen, numReads, numMetrics
    cdef tmplSeq_t polishedTmplSeq = np.zeros(tmplSeq.shape[0], dtype=np.uint8)
    
    cdef np.float64_t[:] results = np.zeros(8 * tmplSeq.shape[0] * readSeqs.shape[1], dtype=np.float64)
    
    readLen = readSeqs.shape[0]
    numReads = readSeqs.shape[1]
    #numMetrics = qvInfo.shape[0]
    numMetrics = 5
    
    run_bqcuda(&tmplSeq[0], &polishedTmplSeq[0], &readSeqs[0, 0], &qvInfo[0, 0, 0], &results[0], tmplScore, realTmplLen, readLen, numReads, numMetrics)
    
    return np.asarray(results), np.asarray(polishedTmplSeq)