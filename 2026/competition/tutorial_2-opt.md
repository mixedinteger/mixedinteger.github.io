---
layout: default
title: GPU-Accelerated 2-Opt Local Search for MIP in Cuda
---

# GPU-Accelerated 2-Opt Local Search for MIP in Cuda

Akif Çördük (acoerduek@nvidia.com) and the 2026 Land-Doig MIP Competition Committee

## 1. Algorithmic Overview
This implementation performs a **2-Opt local search** on a Mixed Integer Programming (MIP) problem using CUDA. The objective is to identify a pair of variables $(x_i, x_j)$ and a corresponding perturbation vector $(\delta_i, \delta_j) \in \{-1, 0, 1\}^2$ such that the objective function value minimizes, subject to linear constraints and variable bounds.

Formally, for a given solution vector $x$, constraint matrix $A$, and bounds $lb, ub$, the algorithm seeks indices $i, j$ and scalars $d_i, d_j$ satisfying:

1.  **Objective Improvement:** $\Delta_{obj} = c_i d_i + c_j d_j < \Delta_{best}$
2.  **Variable Bounds:** $lb_k \le x_k + d_k \le ub_k$ for $k \in \{i, j\}$
3.  **Linear Feasibility:** $A(x + \delta) \le b$

Due to the computational complexity of checking $O(N^2)$ pairs against $M$ constraints, the workload is offloaded to the GPU using a hybrid data representation to maximize memory throughput.

---

## 2. Hybrid Data Representation
To balance memory bandwidth efficiency with computational complexity, the system employs a dual representation of the constraint matrix $A$.

### A. Sparse Representation (Compressed Sparse Column - CSC)
The matrix is stored in CSC format (`col_ptr`, `row_ind`, `val`).
* **Purpose:** Efficient iteration over active constraints.
* **Advantage:** When evaluating a perturbation of variable $x_i$, the kernel iterates only over the non-zero rows associated with column $i$. This reduces the complexity of feasibility checks from $O(M)$ to $O(nnz(col_i))$.
* **Memory Access:** Because threads in a block access contiguous columns, the `col_ptr` reads are coalesced. However, accessing `row_ind` and `val` results in gathered reads, which is mitigated by the dense lookup described below.

### B. Dense Representation (Row-Major Linear Array)
A full, flattened copy of matrix $A$ is stored in global memory.
* **Purpose:** $O(1)$ intersection checks and coefficient lookups.
* **Advantage:** When processing column $i$, the algorithm frequently requires the coefficient of variable $j$ in the same row ($A_{row, j}$). Searching for $j$ in the sparse structure of row $i$ would incur $O(\log N)$ or $O(N)$ overhead. The dense matrix allows for immediate random access.
* **Scalability Note:** While efficient for runtime, this approach is memory-bound ($O(N \cdot M)$). For large-scale instances, this structure should be replaced by a hash map or bloom filter.

---

## 3. Parallel Execution Model
The kernel `find_2opt_move_kernel_hybrid` utilizes a coarse-grained parallelization strategy where the problem space is decomposed by variable pairs.

### A. Grid and Block Topology
* **Grid:** A 2D grid allows naturally mapping pairs $(i, j)$ to CUDA blocks.
* **Block:** Each CUDA block is responsible for evaluating exactly one pair $(i, j)$.
* **Threads:** The threads within a block ($T_{block} = 128$) cooperate to verify the feasibility of the constraints affected by the perturbation. If constraints are small on average, the block size could be reduced.

### B. Execution Flow (Per Block)
1.  **Perturbation Space Exploration:** The block iterates through the 8 non-zero permutations of $d_i, d_j \in \{-1, 0, 1\}$.
2.  **Bound and Objective Pruning:** Before checking linear constraints, the thread logic verifies variable bounds and computes the potential objective delta. If the potential improvement is inferior to the current global best, the computationally expensive feasibility check is skipped (Pruning).
3.  **Parallel Feasibility Check:**
    * **Differential checking:** The kernel utilizes a precomputed `activity` vector ($Ax$). The feasibility condition becomes $activity_{row} + \Delta_{row} \le b_{row}$.
    * **Phase 1 (Column $i$):** Threads iterate in parallel over the non-zero rows of column $i$. If column $j$ also has a non-zero in the current row (checked via Dense lookup), the combined effect is calculated.
    * **Phase 2 (Column $j$):** Threads iterate over non-zero rows of column $j$. To avoid double-counting, rows containing non-zeros for *both* $i$ and $j$ are skipped here, as they were handled in Phase 1.
    * **Synchronization:** A boolean flag in shared memory (`s_feasible`) is used. If any thread detects a violation, the flag is lowered. `__syncthreads()` ensures all threads agree on feasibility before proceeding.

### C. Critical Section and Synchronization
The kernel employs `cuda::atomic_ref` (from the CUDA C++ Core Libraries / CCCL) to manage concurrent updates to the global result structure.
* **Optimistic Read:** The global best objective is read without locking to allow early exit.
* **Mutex Pattern:** If a move is feasible and superior, the thread attempts to acquire a spin-lock on `d_result->mutex`.
* **Atomic Exchange:** Uses `atomic_ref::exchange` with `memory_order_acquire` and `memory_order_release` semantics to ensure thread safety when updating the best solution found.

---

## 4. Host-Side Implementation
The `main` function serves as the driver:
1.  **Preprocessing:** Converts the dense host input into the required CSC format.
2.  **Activity Calculation:** Pre-calculates the constraint activity $A \cdot x_0$ to enable differential checking on the GPU.
3.  **Memory Management:** Utilizes `thrust::device_vector` for RAII-compliant memory allocation and host-to-device transfers.
4.  **Result Application:** Retrieves the best move and applies the delta to the host solution vector.

---

## 5. Optimization Analysis and Future Work

### Current Optimizations
* **Cooperative Constraint Checking:** Distributing the traversal of constraint rows across threads within a block maximizes memory bandwidth utilization compared to a single-thread-per-pair approach.
* **Differential Updates:** Computing only the change ($\Delta$) in activity rather than recomputing the full dot product reduces floating-point operations.

### Proposed Optimizations
1.  **Hierarchical Reduction:** Currently, the first thread to find a better move locks the global mutex. A more scalable approach would be to maintain a **thread-local best**, perform a **block-wide reduction** in shared memory, and only perform one global atomic operation per block.
2.  **Stream Compaction / Two-Stage Reduction:** Instead of updating a single global variable, blocks could write valid moves to a global buffer. A secondary `thrust::reduce` or custom kernel could then identify the global minimum. This eliminates contention on the global mutex.
3.  **Work Coarsening:** Assigning multiple variable pairs to a single block could improve instruction-level parallelism (ILP) and amortize the cost of block initialization, though register pressure must be monitored.

---

## 6. Code

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat>
#include <cuda/atomic> // Required for CCCL atomic_ref


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
     if (abort) exit(code);
  }
}


struct MoveResult {
   int mutex;       // 0 = unlocked, 1 = locked
   float delta_obj;
   int i;          
   int j;          
   int di;         
   int dj;         
};


/**
* CUDA Kernel: Hybrid Sparse/Dense 2-Opt Check
* - One Block per pair (i, j)
* - Uses Sparse CSC to iterate active constraints.
* - Uses Dense Matrix for O(1) coefficient lookups and intersection checks.
* - Uses CCCL atomic_ref for efficient global synchronization (wait/notify).
*/
__global__ void find_2opt_move_kernel_hybrid(
   const int* __restrict__ d_A_col_ptr,
   const int* __restrict__ d_A_row_ind,
   const float* __restrict__ d_A_val,
   const float* __restrict__ d_A_dense, // PRECOMPUTED DENSE LOOKUP
   const float* __restrict__ d_b,
   const float* __restrict__ d_c,
   const float* __restrict__ d_x,
   const float* __restrict__ d_lb,
   const float* __restrict__ d_ub,
   const float* __restrict__ d_activity,
   int num_vars,
   MoveResult* d_result
) {
   int i = blockIdx.x;
   int j = blockIdx.y;


   if (i >= num_vars || j >= num_vars || i >= j) return;


   // Get Column Ranges from Global Memory
   int start_i = d_A_col_ptr[i];
   int end_i   = d_A_col_ptr[i+1];


   int start_j = d_A_col_ptr[j];
   int end_j   = d_A_col_ptr[j+1];


   __shared__ int s_feasible;


   float c_i = d_c[i];
   float c_j = d_c[j];
   float x_i = d_x[i];
   float x_j = d_x[j];


   // Loop through perturbations
   for (int di = -1; di <= 1; ++di) {
       float new_xi = x_i + (float)di;
       if (new_xi < d_lb[i] || new_xi > d_ub[i]) continue;


       for (int dj = -1; dj <= 1; ++dj) {
           if (di == 0 && dj == 0) continue;
           float new_xj = x_j + (float)dj;
           if (new_xj < d_lb[j] || new_xj > d_ub[j]) continue;


           float potential_delta = c_i * (float)di + c_j * (float)dj;
          
           // Optimization: Read strictly to avoid overhead, but careful with stale data
           if (potential_delta >= d_result->delta_obj) continue;


           // --- Feasibility Check ---
           if (threadIdx.x == 0) s_feasible = 1;
           __syncthreads();


           // 1. Iterate constraints in Column I
           if (di != 0) {
               int count_i = end_i - start_i;
               for (int k = threadIdx.x; k < count_i; k += blockDim.x) {
                   int idx = start_i + k;
                   // !!Coalesced reads!!
                   //  Each neighboring thread in the block is accessing neighboring elements.
                   int row = d_A_row_ind[idx];
                   float val_i = d_A_val[idx];
                  
                   float val_j = 0.0f;
                   if (dj != 0) {
                       val_j = d_A_dense[row * num_vars + j];
                   }


                   float change = val_i * (float)di + val_j * (float)dj;
                  
                   if (d_activity[row] + change > d_b[row] + 1e-5f) {
                       s_feasible = 0;
                   }
               }
           }
           __syncthreads();
           if(s_feasible == 0) {
             continue;
           }
           // 2. Iterate constraints in Column J
           if (dj != 0) {
               int count_j = end_j - start_j;
               for (int k = threadIdx.x; k < count_j; k += blockDim.x) {
                   if (s_feasible == 0) break;


                   int idx = start_j + k;
                   // coalesced read!
                   int row = d_A_row_ind[idx];


                   if (di != 0) {
                       if (d_A_dense[row * num_vars + i] != 0.0f) {
                           continue;
                       }
                   }
                   // coalesced read
                   float val_j = d_A_val[idx];
                   float change = val_j * (float)dj;


                   if (d_activity[row] + change > d_b[row] + 1e-5f) {
                       s_feasible = 0;
                   }
               }
           }
          
           __syncthreads();


           if (threadIdx.x == 0 && s_feasible) {
               // --- CRITICAL SECTION WITH CCCL ATOMIC_REF ---
               // 1. Check if we are potentially better (Optimization: avoids unnecessary locking)
               if (potential_delta < d_result->delta_obj) {
                   // Create an atomic reference to the mutex in global memory
                   // Scope: Device (visible to all threads on the GPU)
                   cuda::atomic_ref<int, cuda::thread_scope_device> lock(d_result->mutex);


                   // 2. Acquire Lock
                   // Try to exchange 0 -> 1.
                   // If it returns 1, the lock is held, so we WAIT.
                   // 'wait(1)' puts the thread to sleep efficiently if the value is 1.
                   while (lock.exchange(1, cuda::std::memory_order_acquire) != 0) {
                       lock.wait(1, cuda::std::memory_order_relaxed);
                   }


                   // 3. Double-Check inside lock (Required because another thread might have updated while we waited)
                   if (potential_delta < d_result->delta_obj) {
                       d_result->delta_obj = potential_delta;
                       d_result->i = i;
                       d_result->j = j;
                       d_result->di = di;
                       d_result->dj = dj;
                   }


                   // 4. Release Lock
                   lock.store(0, cuda::std::memory_order_release);
                  
                   // Wake up any threads sleeping in 'wait()'
                   lock.notify_all();
               }
           }
           // wait so that next iteration waits until the current iteration results are written
           __syncthreads();
       }
   }
}


int main() {
   const int N = 3; // Vars
   const int M = 2; // Constraints


   std::vector<float> h_c = {-2.0f, -3.0f, -4.0f};
   std::vector<float> h_b = {4.0f, 3.0f};
  
   std::vector<float> h_A_dense = {
       3.0f, 2.0f, 1.0f,
       1.0f, 1.0f, 2.0f 
   };


   // --- Convert to CSC ---
   std::vector<int> h_col_ptr = {0};
   std::vector<int> h_row_ind;
   std::vector<float> h_val;


   for (int col = 0; col < N; ++col) {
       for (int row = 0; row < M; ++row) {
           float val = h_A_dense[row * N + col];
           if (abs(val) > 1e-6) {
               h_row_ind.push_back(row);
               h_val.push_back(val);
           }
       }
       h_col_ptr.push_back(h_row_ind.size());
   }


   std::vector<float> h_lb(N, 0.0f);
   std::vector<float> h_ub(N, 1.0f);
   std::vector<float> h_x = {0.0f, 0.0f, 0.0f};
  
   std::vector<float> h_activity(M, 0.0f);
   for(int k=0; k<M; k++) {
       for(int i=0; i<N; i++) {
           h_activity[k] += h_A_dense[k*N + i] * h_x[i];
       }
   }


   // --- Allocate and Copy using Thrust ---
   // Thrust handles allocation on constructor and copy via assignment
   thrust::device_vector<int> d_A_col_ptr = h_col_ptr;
   thrust::device_vector<int> d_A_row_ind = h_row_ind;
   thrust::device_vector<float> d_A_val = h_val;
   thrust::device_vector<float> d_A_dense = h_A_dense;
  
   thrust::device_vector<float> d_b = h_b;
   thrust::device_vector<float> d_c = h_c;
   thrust::device_vector<float> d_x = h_x;
   thrust::device_vector<float> d_lb = h_lb;
   thrust::device_vector<float> d_ub = h_ub;
   thrust::device_vector<float> d_activity = h_activity;


   // Initialize result on device
   MoveResult initial_res = {0, 0.0f, -1, -1, 0, 0};
   thrust::device_vector<MoveResult> d_result(1, initial_res);


   // --- Launch Kernel ---
   dim3 grid(N, N);
   int threadsPerBlock = 128;


   std::cout << "Launching Hybrid Kernel using Thrust..." << std::endl;
  
   // Use raw_pointer_cast to extract pointers for the kernel
   find_2opt_move_kernel_hybrid<<<grid, threadsPerBlock>>>(
       thrust::raw_pointer_cast(d_A_col_ptr.data()),
       thrust::raw_pointer_cast(d_A_row_ind.data()),
       thrust::raw_pointer_cast(d_A_val.data()),
       thrust::raw_pointer_cast(d_A_dense.data()),
       thrust::raw_pointer_cast(d_b.data()),
       thrust::raw_pointer_cast(d_c.data()),
       thrust::raw_pointer_cast(d_x.data()),
       thrust::raw_pointer_cast(d_lb.data()),
       thrust::raw_pointer_cast(d_ub.data()),
       thrust::raw_pointer_cast(d_activity.data()),
       N,
       thrust::raw_pointer_cast(d_result.data())
   );
   cudaCheckError(cudaDeviceSynchronize());


   // --- Results ---
   MoveResult best_move;
   cudaMemcpy(&best_move, thrust::raw_pointer_cast(d_result.data()), sizeof(MoveResult), cudaMemcpyDeviceToHost);


   std::cout << "Best Move Found:\n";
   std::cout << "  Obj Delta: " << best_move.delta_obj << "\n";
  
   if (best_move.delta_obj < 0) {
       std::cout << "  Apply: x[" << best_move.i << "] += " << best_move.di << "\n";
       std::cout << "  Apply: x[" << best_move.j << "] += " << best_move.dj << "\n";
       h_x[best_move.i] += best_move.di;
       h_x[best_move.j] += best_move.dj;
       std::cout << "  New Solution Vector: [ ";
       for(float val : h_x) std::cout << val << " ";
       std::cout << "]" << std::endl;
   }


   return 0;
}
```
