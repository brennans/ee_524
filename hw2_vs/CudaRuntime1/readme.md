# Homework 2

## Questions

1.  Shared memory is per block and so there will be a variable for each thread block.
    b. 1000
2.  Originally, each thread was loading a row for M and a column for N.  If we use a 32x32 tile, 
    the threads in the tile can share the 32 elements in the row and the column that they need 
    to access.  So for every 32 accesses we had originally we now only need 1.
    c. 1/32 of the original usage.
3. 32 x 32 = 1024 elements = 2048 bytes.  2048 bytes / 128 bytes per DRAM burst = 16 DRAM bursts.
   (Note this is probably a simplified calculation due to the accesses working out to an integer
    value.)
    a. 16


Nsight VSE Debug:
[warp_lane_debug.png]  There are 1280 thread which would mean there are 40 warps 
running in this screenshot.  This was taken on a personal computer with an 8.6 Compute 
Capability GPU which is under the maximum of 48 warps per SM on this architecture.  I think on the 
7.5 Compute Capability GPU this would be over the limit of 32 warps per SM.

```
>ncu -k blurKernelGlobal --metrics smsp__cycles_elapsed.sum,smsp__cycles_elapsed.sum.per_second,smsp__inst_executed.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes.sum CudaRuntime1.exe
Running Global Kernel
==PROF== Connected to process 21252 (C:\Users\brenn\workspace\ee_524\hw2_vs\CudaRuntime1\CudaRuntime1.exe)
Processing data/hw2_testimage1.png
Image Parameters. Size: 57600, Width: 240, Height: 240
Using Kernel Parameters...
Block Dimensions: (8, 8)     Thread Dimensions: (32, 32)
==PROF== Profiling "blurKernelGlobal": 0%....50%....100% - 1 pass
Total computation time value: 462837500ns, Windows: 462837 us
GPU kernel time: 187910200ns, Windows: 187910 us
==PROF== Disconnected from process 21252
[21252] CudaRuntime1.exe@127.0.0.1
blurKernelGlobal(unsigned char *, const unsigned char *, const float *, int, int), 2022-Oct-24 21:34:00, Context 1, Stream 7
Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
dram__bytes.sum                                                                  Kbyte                         542.98
dram__sectors_read.sum                                                          sector                         14,988
dram__sectors_write.sum                                                         sector                          1,980
smsp__cycles_elapsed.sum                                                         cycle                      6,256,536
smsp__cycles_elapsed.sum.per_second                                      cycle/nsecond                         124.61
smsp__inst_executed.sum                                                           inst                      2,449,280
---------------------------------------------------------------------- --------------- ------------------------------
```

```
>ncu -k blurKernelGlobal --metrics smsp__cycles_elapsed.sum,smsp__cycles_elapsed.sum.per_second,smsp__inst_executed.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes.sum CudaRuntime2.exe
Running Global Kernel
==PROF== Connected to process 21540 (C:\Users\brenn\workspace\ee_524\hw2_vs\CudaRuntime1\CudaRuntime2.exe)
Processing data/hw2_testimage2.png
Image Parameters. Size: 262144, Width: 512, Height: 512
Using Kernel Parameters...
Block Dimensions: (17, 17)     Thread Dimensions: (32, 32)
==PROF== Profiling "blurKernelGlobal": 0%....50%....100% - 1 pass
Total computation time value: 434862700ns, Windows: 434862 us
GPU kernel time: 176924400ns, Windows: 176924 us
==PROF== Disconnected from process 21540
[21540] CudaRuntime2.exe@127.0.0.1
blurKernelGlobal(unsigned char *, const unsigned char *, const float *, int, int), 2022-Oct-24 21:34:05, Context 1, Stream 7
Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
dram__bytes.sum                                                                  Mbyte                           2.69
dram__sectors_read.sum                                                          sector                         67,688
dram__sectors_write.sum                                                         sector                         16,292
smsp__cycles_elapsed.sum                                                         cycle                     20,674,776
smsp__cycles_elapsed.sum.per_second                                      cycle/nsecond                         124.75
smsp__inst_executed.sum                                                           inst                     10,455,360
---------------------------------------------------------------------- --------------- ------------------------------
```

```
>ncu -k blurKernelGlobal --metrics smsp__cycles_elapsed.sum,smsp__cycles_elapsed.sum.per_second,smsp__inst_executed.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes.sum CudaRuntime3.exe
Running Global Kernel
==PROF== Connected to process 13784 (C:\Users\brenn\workspace\ee_524\hw2_vs\CudaRuntime1\CudaRuntime3.exe)
Processing data/hw2_testimage3.png
Image Parameters. Size: 722500, Width: 850, Height: 850
Using Kernel Parameters...
Block Dimensions: (27, 27)     Thread Dimensions: (32, 32)
==PROF== Profiling "blurKernelGlobal": 0%....50%....100% - 1 pass
Total computation time value: 426790500ns, Windows: 426790 us
GPU kernel time: 176133100ns, Windows: 176133 us
==PROF== Disconnected from process 13784
[13784] CudaRuntime3.exe@127.0.0.1
blurKernelGlobal(unsigned char *, const unsigned char *, const float *, int, int), 2022-Oct-24 21:39:58, Context 1, Stream 7
Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
dram__bytes.sum                                                                  Mbyte                           6.33
dram__sectors_read.sum                                                          sector                        175,484
dram__sectors_write.sum                                                         sector                         22,464
smsp__cycles_elapsed.sum                                                         cycle                     53,233,008
smsp__cycles_elapsed.sum.per_second                                      cycle/nsecond                         124.78
smsp__inst_executed.sum                                                           inst                     29,265,030
---------------------------------------------------------------------- --------------- ------------------------------
```

```
>ncu -k blurKernelGlobal --metrics smsp__cycles_elapsed.sum,smsp__cycles_elapsed.sum.per_second,smsp__inst_executed.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes.sum CudaRuntime4.exe
Running Global Kernel
==PROF== Connected to process 10480 (C:\Users\brenn\workspace\ee_524\hw2_vs\CudaRuntime1\CudaRuntime4.exe)
Processing data/hw2_testimage4.png
Image Parameters. Size: 12192768, Width: 4032, Height: 3024
Using Kernel Parameters...
Block Dimensions: (127, 95)     Thread Dimensions: (32, 32)
==PROF== Profiling "blurKernelGlobal": 0%....50%....100% - 1 pass
Total computation time value: 514707900ns, Windows: 514707 us
GPU kernel time: 202871700ns, Windows: 202871 us
==PROF== Disconnected from process 10480
[10480] CudaRuntime4.exe@127.0.0.1
blurKernelGlobal(unsigned char *, const unsigned char *, const float *, int, int), 2022-Oct-24 21:40:03, Context 1, Stream 7
Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
dram__bytes.sum                                                                  Mbyte                          60.49
dram__sectors_read.sum                                                          sector                      1,509,656
dram__sectors_write.sum                                                         sector                        380,656
smsp__cycles_elapsed.sum                                                         cycle                    852,543,440
smsp__cycles_elapsed.sum.per_second                                      cycle/nsecond                         124.80
smsp__inst_executed.sum                                                           inst                    488,471,920
---------------------------------------------------------------------- --------------- ------------------------------
```

       (Nanoseconds)  Global Timer 1, Global Timer 1, Static Timer 1, Static Timer 2, Dynamic Timer 1, Dynamic Timer 2,     Host Timer
../hw2_outimage1.png:       84415500,          78400,         592400,          60100,         773400,         117900,      162314100
../hw2_outimage2.png:         773900,         206400,         683200,         177600,         744700,         202000,      746529700
../hw2_outimage3.png:        4896300,         361900,         979800,         290100,         998100,         360800,     2056586200
