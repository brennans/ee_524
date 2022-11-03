# Homework 3

## Questions

1. Using 128 threads per block would be limited by the number of blocks to only 512 threads. The same issue would occur with 256 threads per block with only 1024 total threads. With 512 threads per block and a total of three blocks, all threads on the SM could be utilized. If the number of threads per block was 1024, then only a single block could be used, and the number of threads would only be 1024 threads.
   Answer. C. 512 threads per block

2. Using 4 blocks of 512 threads each would result in 2048 threads.
   C. 2048

3. Each warp of 32 threads, would result in only one warp having divergence.  While there would be two warps that fail the boudary check, one would have all threads fail, and so not have divergence, but the other would have 16 threads fail, and 16 threads pass the check.
   A. 1

4. A. 8 blocks with 128 threads would result in 1024 threads. Possible.
   B. 16 blocks with 64 threads would result in 1024 threads. Possible.
   C. 32 blocks with 32 threads would result in 1024 threads. Possible.
   D. 64 blocks with 32 threads would result in 2048 threads. Possible.
   E. 32 blocks with 64 threads would result in 2048 threads. Possible.

5. A. 128 threads per block, would need 16 blocks for full occupancy, and a total of 61440 registers which would be under the device limit.
   B. 32 threads per block, would need 64 blocks for full occupancy, and so could only acheive 50% occupancy because of the 32 block limit.
   C. 256 threads per block would need 8 blocks for full occupancy, but would be limited by the number of registers as 69632 registers would be required for this many threads which is over the 64K registers the device supports.


## Control Divergence Analysis

Condition: (p * TILE_WIDTH + ty < Width)

Assumptions:
    - 16x16 tiles and thread blocks (256 threads per block)
    - Each thread block has 8 warps (32 threads per warp)
    - Inputs are 100 x 100 square matrices

There are a total of 49 blocks.  For 42 of these blocks (Type 1), each block runs a total 7 times with 6 of those times not having any control divergence. And the lasttime, would also not have any control divergence as each full warp would either have a valid index in all thread, or invalid indicies in all threads. this is a total of (8 warps * 42 blocks * 7 phases) = 2352 warp-phases without divergence. 
Type 1 blocks: 0 / 2352 Warp phases have control divergence, No performance impact.

For the 7 blocks where there is a not a full tile of threads (Type 2), the first 6 iterations, would have control divergence in all 8 warps. (8 warps * 7 blocks * 6 iterations) would result in 336 warp-phases with divergence. The last iteration, would have only 4 warps with divergence and the remaining would all not have valid indicies. There are are total of (8 warps * 7 blocks * 7 iterations) = 392 warp-phases for Type 2 block.
Type 2 blocks: 340 / 392 warp phases have control divergence. 86.7% of warp-phases.

Total: 340 / 2734 warp-phases have control divergence. 12.44% of warp-phases.
