# Homework #1

Brennan Swanton

----------------

## Answers to Questions

### Part 1

1. C. `i=blockIdx.x*blockDim.x + threadIdx.x;`

2. C. `i=(blockIdx.x*blockDim.x + threadIdx.x)*2`

3. D. `i=blockIdx.x*blockDim.x*2 + threadIdx.x`

4. C. 8192
```
ans = ceil(8000 / 1024) * 1024
```


### Part 2

#### printf kernel

4. The printf results from the threads within a block indicate that the threads are running 
   sequentially after each other as the thread ids are always in order from 0-3, but the block 
   index is not in sequential order so they could run in any order. It is probably not safe to 
   assume that threads or blocks are scheduled in any particular order.

#### Host-Side Result Verification

1. For the SAXPY kernel, results will fail on larger values at N = 1e7. This can require a tolerance
value of up to 10.  The 3D matrix works for values that are equal to indexes down to 
   std::numeric_limit<float>::min(). The SAXPY has the most severe result disagreements likely due
   to the multiplication.