#undef _GLIBCXX_USE_INT128
#undef _GLIBCXX_ATOMIC_BUILTINS

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/extrema.h>

#include "bqcuda.h"

#define cudaSafeCall(call){   \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

#define MAX_THREADS_PER_BLOCK 1024
#define SCORE_THREADS_PER_BLOCK 256
#define TALLY_THREADS_PER_BLOCK 768
#define MAX_BLOCKS_PER_GRID 65535

//[C2.AllQVsModel]
#define Match              0.2627555
#define Mismatch          -1.09688872
#define MismatchS         -0.01637988
#define Branch            -0.60275947
#define BranchS           -0.02682689
#define DeletionN         -1.00012494
#define DeletionWithTag    0.06000148
#define DeletionWithTagS  -0.02579358
#define Nce               -0.15864559
#define NceS              -0.04403654
#define Merge             -1.02398814
#define MergeS            -0.12135255

//metrics
#define InsertionIdx 0
#define MergeIdx 1
#define DeletionIdx 2
#define DeletionTagIdx 3
#define SubstitutionIdx 4

//https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

__host__ __device__ void indexToMutation(uint8_t *origTmplSeq, int tmplLen, uint8_t *mutated, int index) {

    int baseOrd[4] = {'A', 'C', 'G', 'T'};
    int tmpPos = index / 8;
    int mut = index % 8;
    int currentBasePos;

    for (int i = 0; i < tmpPos; i++) {
        mutated[i] = origTmplSeq[i];
    }

    for (int i = 0; i < 4; i++) {
        if (origTmplSeq[tmpPos] == baseOrd[i]) {
            currentBasePos = i;
        }
    }

    if (mut < 4) {
        mutated[tmpPos] = baseOrd[mut % 4];
        for (int i = tmpPos; i < tmplLen - 1; i++) {
            mutated[i + 1] = origTmplSeq[i];
        }
        mutated[tmplLen - 1] = '\0';
    } else if (mut >= 4 and mut < 7) {
        mutated[tmpPos] = baseOrd[(currentBasePos + ((mut-4+1) % 4)) % 4];
        for (int i = tmpPos + 1; i < tmplLen; i++) {
            mutated[i] = origTmplSeq[i];
        }
    } else {
        for (int i = tmpPos + 1; i < tmplLen; i++) {
            mutated[i-1] = origTmplSeq[i];
        }
        mutated[tmplLen - 1] = '\0';
    }

}

__device__ uint8_t ReadBase(uint8_t *readSeqs, size_t rs_pitch, int read_id, int read_pos) {
    return *((uint8_t*)((char*) readSeqs + read_pos * rs_pitch) + read_id);
}

__device__ uint8_t QVInfoVal(cudaPitchedPtr qvInfo, int readLen, int metric, int read_id, int read_pos) {
    size_t pitch = qvInfo.pitch;
    size_t slicePitch = pitch * readLen;
    return *((uint8_t*)(((char*) qvInfo.ptr + metric * slicePitch) + read_pos * pitch) + read_id);
}

__global__ void ScoreMutations(uint8_t *origTmplSeq, uint8_t *readSeqs, size_t rs_pitch, cudaPitchedPtr qvInfo, int tmplLen, int readLen, int numReads, double *results) {

    int block_global_index = SCORE_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
    int thread_id = (blockDim.x * threadIdx.y) + threadIdx.x;
    int global_index = block_global_index + thread_id;
    if (global_index >= 8 * tmplLen * numReads) return;

    int mut_id = global_index / numReads;
    int read_id = global_index % numReads;

    uint8_t tmplSeq[128];

    indexToMutation(origTmplSeq, tmplLen, tmplSeq, mut_id);

    double alpha[4][130];

    double score = -FLT_MAX;
    int lastAlphaI, lastAlphaJ;
    double moveScore = 0;

    for (int i = 0; i < readLen + 1; i++) {

        int alphamodi = i % 4;
        int alphamodim1 = (i - 1) % 4;
        if (i > 0 and ReadBase(readSeqs, rs_pitch, read_id, i - 1) == 0) {
            lastAlphaI = alphamodim1;
            break;
        }

        for (int j = 0; j < tmplLen + 1; j++) {

            if (j > 0 and tmplSeq[j - 1] == 0) {
                lastAlphaJ = j - 1;
                break;
            }

            score = (-FLT_MAX);

            if (i == 0 and j == 0) {
                score = 0;
            }

            moveScore = -1;

            //Incorporate
            if (i > 0 and j > 0) {

                if (ReadBase(readSeqs, rs_pitch, read_id, i - 1) == tmplSeq[j - 1]) {
                    moveScore = alpha[alphamodim1][j - 1] + Match;
                } else {
                    moveScore = alpha[alphamodim1][j - 1] + Mismatch + MismatchS * QVInfoVal(qvInfo, readLen, SubstitutionIdx, read_id,  i - 1);
                }
                score = (moveScore > score) ? moveScore : score;
            }
            //Extra
            if (i > 0) {
                if (j < tmplLen and ReadBase(readSeqs, rs_pitch, read_id, i - 1) == tmplSeq[j]) {
                    moveScore = alpha[alphamodim1][j] + Branch + BranchS * QVInfoVal(qvInfo, readLen, InsertionIdx, read_id,  i - 1);
                } else {
                    moveScore = alpha[alphamodim1][j] + Nce + NceS * QVInfoVal(qvInfo, readLen, InsertionIdx, read_id,  i - 1);
                }
                score = (moveScore > score) ? moveScore : score;
            }

            //Delete
            if (j > 0) {
                if (i < readLen and QVInfoVal(qvInfo, readLen, DeletionTagIdx, read_id,  i) == tmplSeq[j - 1]) {
                    moveScore = alpha[alphamodi][j - 1] + DeletionWithTag + DeletionWithTagS * QVInfoVal(qvInfo, readLen, DeletionIdx, read_id,  i);
                } else {
                    moveScore = alpha[alphamodi][j - 1] + DeletionN;
                }
                score = (moveScore > score) ? moveScore : score;
            }

            //Merge
            if (i > 0 and j > 1) {
                if (! (ReadBase(readSeqs, rs_pitch, read_id, i - 1) == tmplSeq[j - 2] && ReadBase(readSeqs, rs_pitch, read_id, i - 1) == tmplSeq[j - 1])) {
                    //moveScore = alpha[i - 1][j - 2] + (-FLT_MAX);
                    moveScore = -FLT_MAX;
                } else {
                    moveScore = alpha[alphamodim1][j - 2] + Merge + MergeS * QVInfoVal(qvInfo, readLen, MergeIdx, read_id,  i - 1);
                }
                score = (moveScore > score) ? moveScore : score;
            }
            alpha[alphamodi][j] = score;
        }
    }

    results[global_index] = alpha[lastAlphaI][lastAlphaJ];

}



int run_bqcuda(uint8_t *origTmplSeq, uint8_t *polishedTmplSeq, uint8_t *readSeqs, uint8_t *qvInfo, double *results, double origTmplScore, int tmplLen, int readLen, int numReads, int numMetrics) {
    
    uint8_t *d_currentTmplSeq;
    uint8_t *d_readSeqs;
    cudaPitchedPtr d_qvInfo;
    
    size_t rs_pitch;

    int template_mutations = 8 * tmplLen;
    double currentTmplScore = origTmplScore;
    
    /** Copy data to GPU **/

    // Copy original template
    memcpy(polishedTmplSeq, origTmplSeq, tmplLen * sizeof(uint8_t));
    cudaSafeCall( cudaMalloc(&d_currentTmplSeq, tmplLen * sizeof(uint8_t)) );
    
    // Copy read seqs
    cudaSafeCall( cudaMallocPitch(&d_readSeqs, &rs_pitch, numReads * sizeof(uint8_t), readLen * sizeof(uint8_t)) );
    
    for (unsigned int i = 0; i < readLen; i++) {
      cudaSafeCall( cudaMemcpy((uint8_t*)((char*) d_readSeqs + i * rs_pitch), readSeqs + i * numReads, sizeof(uint8_t) * numReads, cudaMemcpyHostToDevice) );
    }

    // Copy QV info
    cudaSafeCall( cudaMalloc3D(&d_qvInfo, make_cudaExtent(numReads * sizeof(uint8_t), readLen, numMetrics)) );
    
    size_t pitch = d_qvInfo.pitch;
    size_t slicePitch = pitch * readLen;
    
    for (int z = 0; z < numMetrics; z++) {
        char* slice = (char*) d_qvInfo.ptr + z * slicePitch;
        for (int y = 0; y < readLen; y++) {
            uint8_t* row = (uint8_t*)(slice + y * pitch);
            cudaSafeCall( cudaMemcpy(row, qvInfo + z * readLen * numReads + y * numReads, sizeof(uint8_t) * numReads, cudaMemcpyHostToDevice) );
        }
    }

    /** Allocate working space on GPU **/

    // Create storage for P(read | mutated template) on GPU
    double *d_results;
    cudaSafeCall( cudaMalloc(&d_results, template_mutations * numReads * sizeof(double)) );
    thrust::device_ptr<double> d_results_start(d_results);
    thrust::device_ptr<double> d_results_end(d_results + template_mutations * numReads);

    // Create storage for P(all reads | mutated template) on GPU
    double *d_mutation_scores;
    cudaSafeCall( cudaMalloc(&d_mutation_scores, template_mutations * sizeof(double)) );
    thrust::device_ptr<double> d_mutation_scores_start(d_mutation_scores);
    thrust::device_ptr<double> d_mutation_scores_end(d_mutation_scores + template_mutations);
    typedef thrust::device_ptr<double> thrust_device_double_ptr;
    strided_range<thrust_device_double_ptr> each_mutation(d_results_start + numReads - 1, d_results_end, numReads);

    // Create temp placeholder space for currentTmplSeq
    uint8_t *copyOfCurrentTmplSeq = (uint8_t*) malloc(tmplLen * sizeof(uint8_t));
    memset(copyOfCurrentTmplSeq, '\0', tmplLen * sizeof(uint8_t));

    /** Score Mutations **/

    // Figure out launch params
    dim3 score_threadsPerBlock(32, 8);

    int score_blocks_needed = (template_mutations * numReads + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;
    int score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
    int score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

    dim3 score_blocksPerGrid(score_block_x, score_block_y);

    // Run it

    int k = 0;

    while (true) {

        cudaSafeCall( cudaMemcpy(d_currentTmplSeq, polishedTmplSeq, tmplLen * sizeof(uint8_t), cudaMemcpyHostToDevice) );
        cudaSafeCall( cudaMemset(d_results, '\0', template_mutations * numReads * sizeof(double)) );
        cudaSafeCall( cudaMemset(d_mutation_scores, '\0', template_mutations * sizeof(double)) );

        ScoreMutations<<<score_blocksPerGrid, score_threadsPerBlock>>>(d_currentTmplSeq, d_readSeqs, rs_pitch, d_qvInfo, tmplLen, readLen, numReads, d_results);

        // Add read log-probs for each mutation to get total prob for that mutation
        thrust::inclusive_scan(d_results_start, d_results_end, d_results_start);
        thrust::adjacent_difference(each_mutation.begin(), each_mutation.end(), d_mutation_scores_start);

        cudaSafeCall( cudaMemcpy(results, d_mutation_scores, template_mutations * sizeof(double), cudaMemcpyDeviceToHost) );
        //results[0] = k;
        // Find max element
        thrust::device_ptr<double> d_max_prob_mutation_pos = thrust::max_element(d_mutation_scores_start, d_mutation_scores_end);

        // If there's a higher scoring mutation, prepare to run again

        if (*d_max_prob_mutation_pos > currentTmplScore) {

            int mut_id = (d_max_prob_mutation_pos - d_mutation_scores_start);

            // Backup current template
            memcpy(copyOfCurrentTmplSeq, polishedTmplSeq, tmplLen * sizeof(uint8_t));
            indexToMutation(copyOfCurrentTmplSeq, tmplLen, polishedTmplSeq, mut_id);
            currentTmplScore = *d_max_prob_mutation_pos;

        } else {

            break;

        }
        k++;
    }

    free(copyOfCurrentTmplSeq);
    cudaSafeCall( cudaFree(d_mutation_scores) );
    cudaSafeCall( cudaFree(d_results) );
    cudaSafeCall( cudaFree(d_qvInfo.ptr) );
    cudaSafeCall( cudaFree(d_readSeqs) );
    cudaSafeCall( cudaFree(d_currentTmplSeq) );

    return 0;
}
