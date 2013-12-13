#ifndef BQCUDA_CUDA
#define BQCUDA_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>

int run_bqcuda(uint8_t *origTmplSeq, uint8_t *polishedTmplSeq, uint8_t *readSeqs, uint8_t *qvInfo, double *results, double origTmplScore, int tmplLen, int readLen, int numReads, int numMetrics);

#ifdef __cplusplus
}
#endif

#endif
