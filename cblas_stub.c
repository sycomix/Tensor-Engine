#include <stddef.h>

// Minimal cblas_sgemm stub to satisfy dynamic linker when no BLAS is available.
// Uses a naive triple loop; parameters use int for enums for ABI compatibility.

void cblas_sgemm(int Order, int TransA, int TransB,
                 int M, int N, int K,
                 float alpha,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float beta,
                 float *C, int ldc) {
    // Initialize C with beta scaling
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * ldc + j;
            C[idx] = C[idx] * beta;
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik;
            if (TransA == 112) { // CblasNoTrans == 112 in cblas headers
                a_ik = A[i * lda + k];
            } else {
                a_ik = A[k * lda + i];
            }
            a_ik *= alpha;
            for (int j = 0; j < N; ++j) {
                float b_kj;
                if (TransB == 112) {
                    b_kj = B[k * ldb + j];
                } else {
                    b_kj = B[j * ldb + k];
                }
                int idx = i * ldc + j;
                C[idx] += a_ik * b_kj;
            }
        }
    }
}
