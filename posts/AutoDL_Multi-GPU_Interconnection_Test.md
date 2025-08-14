---
date: 2025-06-05
category:
  - AI
tag:
  - GPU
  - AutoDL
star: true
sticky: true
description: AutoDL平台2080Ti双卡GPU互联性能测试报告。详细分析PCIe 3.0 x16拓扑结构下P2P通信、NCCL集合通信性能瓶颈，提供多GPU训练环境搭建参考。
keywords: AutoDL, GPU测试, 2080Ti, 多卡互联, PCIe性能, CUDA测试, 深度学习
---

# AutoDL GPU 容器多 GPU 互联测试-北京 A 区-2080Ti x2

因业务需要可能会去 AutoDL 租用服务器用于模型训练，故测试了一下其平台上消费级显卡（原版2080Ti 11G）多卡互联时的性能表现。
将结果放在这里作为参考。

省流：没有NVLink，虽然运行在PCIe 3.0x16交换机上，且两个CPU NUMA都接入了PCIe switch，但由于驱动原因，P2P通信不可用，导致多卡互联性能较差。

> 以下为 DeepSeek-R1-0528 的总结
-------
#### 一、测试环境概览

1. **硬件配置**：
    
    - GPU：NVIDIA GeForce RTX 2080 Ti × 2（各 11GB 显存）
    - 拓扑结构：PCIe Gen3 ×16 互联（`nvidia-smi topo` 显示 `PIX` 模式）
    - NUMA 亲和性：双卡共享同一 NUMA 节点（Node 1）
2. **软件环境**：
    
    - CUDA 版本：12.4
    - 驱动版本：550.90.07
    - 测试工具：`cuda-samples`、`nccl-tests`

#### 二、关键测试结果分析

##### 1. **GPU 基础状态**

- **负载表现**（`nvtop`）：
    - 双卡持续高负载（GPU 利用率 ≈100%）
    - 显存占用稳定在 3.48GB/11GB（约 31.6%）
    - 功耗：91-92W（低于 TDP 250W）
- **互联拓扑**（`nvidia-smi topo`）：
    - 不支持 NVLink（`NS` 状态）
    - 仅通过 PCIe Gen3 ×16 互联（理论带宽双向 32GB/s）

##### 2. **点对点通信性能**

- **`p2pBandwidthLatencyTest`**：
    - **P2P 访问不可用**：`Device=0 CANNOT Access Peer Device=1`
    - 单向带宽：
        - GPU0 → GPU1：5.71 GB/s
        - GPU1 → GPU0：5.70 GB/s
    - 双向带宽：
        - P2P 禁用时：6.10–6.12 GB/s
        - P2P 启用时：6.10 GB/s（无提升）
    - 通信延迟：14.76 μs（GPU0→GPU1）

> 💡 **结论**：PCIe 互联带宽利用率不足理论值 20%，且 P2P 加速无效。

##### 3. **NCCL 集合通信性能**

| **操作类型**     | **最大带宽 (busbw)** | **瓶颈分析**              |
| ------------ | ---------------- | --------------------- |
| `all_reduce` | 4.00 GB/s        | 受限于 PCIe 双向带宽         |
| `broadcast`  | 11.09 GB/s       | 接近 PCIe 单向理论峰值（69.4%） |
| `reduce`     | 11.14 GB/s       | 接近 PCIe 单向理论峰值（69.6%） |
| `all_gather` | 7.69 GB/s        | 数据分发效率低于单向操作          |
| `alltoall`   | 9.45 GB/s        | 多对多通信优化效果显著           |

- **带宽规律**：
    - 单向操作（broadcast/reduce）> 多向操作（alltoall）> 全局聚合（all_reduce）
    - 小数据包（<1MB）带宽骤降（最低 0.01 GB/s）




> 以下为原始测试数据
-------

### nvcc -V
```bash
root@autodl-container-xxxxx:~# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

### nvidia-smi
```bash
root@autodl-container-xxxxx:~# nvidia-smi
Thu Jun  5 17:15:31 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:B4:00.0 Off |                  N/A |
| 29%   31C    P8             20W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:B5:00.0 Off |                  N/A |
| 29%   31C    P8             20W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### nvtop

```bash
root@autodl-container-xxxxx:~# nvtop

 Device 0 [NVIDIA GeForce RTX 2080 Ti] PCIe GEN 3@16x RX: 4.815 MiB/s TX: 5.496 MiB/s
 GPU 1905MHz MEM 6800MHz TEMP  43°C FAN  35% POW  91 / 250 W
 GPU[||||||||||||||||||||||||||||||||100%] MEM[|||||||||||          3.483Gi/11.000Gi]

 Device 1 [NVIDIA GeForce RTX 2080 Ti] PCIe GEN 3@16x RX: 5.112 MiB/s TX: 5.615 MiB/s
 GPU 1905MHz MEM 6800MHz TEMP  40°C FAN  35% POW  92 / 250 W
 GPU[||||||||||||||||||||||||||||||||100%] MEM[|||||||||||          3.483Gi/11.000Gi]
   ┌────────────────────────────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────┐
100│GPU0 %      ┌─────────────┐                       ┌─────────│100│GPU1 %      ┌─────────────┐                       ┌─────────│
   │GPU0 mem% ┌─┘             │                     ┌─┘         │   │GPU1 mem% ┌─┘             │                     ┌─┘         │   
   │          │               └─┐                   │           │   │          │               │                     │           │   
   │          │                 │                   │           │   │          │               │                   ┌─┘           │ 
 75│          │                 │                 ┌─┘           │ 75│          │               │                   │             │   
   │          │                 │                 │             │   │          │               │                   │             │
   │          │                 │                 │             │   │          │               │                   │             │   
   │          │                 │               ┌─┘             │   │          │               │                   │             │
 50│          │                 │               │               │ 50│          │               │                   │             │   
   │          │                 │               │               │   │          │               │                   │             │   
   │          │┌───────────────┐│              ┌┼───────────────│   │          │┌──────────────┼┐               ┌──┼─────────────│   
   │          ││               ││              ││               │   │          ││              └┼┐              │  │             │ 
 25│          ││               ││              ││               │ 25│          ││               ││              │┌─┘             │   
   │          ││               ││              ││               │   │          ││               ││              ││               │   
   │          ││               │└─┐            ││               │   │          ││               ││              ││               │  
  0│──────────┴┘               └──┴────────────┴┘               │  0│──────────┴┘               └┴──────────────┴┘               │   
   └────────────────────────────────────────────────────────────┘   └────────────────────────────────────────────────────────────┘    
   PID USER DEV    TYPE  GPU        GPU MEM    CPU  HOST MEM Command                                                              
   124949  N/A   0 Compute  99%   3304MiB  29%    N/A       N/A                                                                      
   124949  N/A   1 Compute  99%   3304MiB  29%    N/A       N/A

F2Setup   F6Sort    F9Kill    F10Quit    F12Save Config
```

### nvidia-smi topo

```bash
root@autodl-container-xxxxx:~# nvidia-smi topo -m
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PIX     24-47,72-95     1               N/A
GPU1    PIX      X      24-47,72-95     1               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

root@autodl-container-xxxxx:~# nvidia-smi topo -p2p p
        GPU0    GPU1
 GPU0   X       NS
 GPU1   NS      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

### cuda-samples: 1_Utilities/bandwidthTest

```bash
root@autodl-container-xxxxx:~/cuda-samples/Samples/1_Utilities/bandwidthTest# ./bandwidthTest  -g 2
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 2080 Ti
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     12.2

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     13.2

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     518.9

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

### cuda-samples: 5_Domain_Specific/p2pBandwidthLatencyTest

```bash
root@autodl-container-xxxxx:~/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest# ./p2pBandwidthLatencyTest
[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA GeForce RTX 2080 Ti, pciBusID: b4, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA GeForce RTX 2080 Ti, pciBusID: b5, pciDeviceID: 0, pciDomainID:0
Device=0 CANNOT Access Peer Device=1
Device=1 CANNOT Access Peer Device=0

***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

P2P Connectivity Matrix
     D\D     0     1
     0       1     0
     1       0     1
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 541.84   5.71
     1   5.70 543.05
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1
     0 538.62   5.72
     1   5.72 532.54
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 533.91   6.12
     1   6.10 533.91
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 533.34   6.10
     1   6.10 534.01
P2P=Disabled Latency Matrix (us)
   GPU     0      1
     0   1.33  14.76
     1  13.95   1.26

   CPU     0      1
     0   3.55  11.08
     1  10.48   3.32
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1
     0   1.32  14.99
     1  13.63   1.25

   CPU     0      1
     0   3.36  10.34
     1  10.47   3.27

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

### nccl-tests: all_reduce_perf, all_gather_perf, broadcast_perf, reduce_perf, alltoall_perf

```bash
root@autodl-container-xxxxx:~/nccl-tests# ./build/all_reduce_perf -b 8 -e 1024M -f 2 -g2
# nThread 1 nGpus 2 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31706 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#  Rank  1 Group  0 Pid  31706 on autodl-container-xxxxx device  1 [0000:b5:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    12.25    0.00    0.00      0    12.00    0.00    0.00      0
          16             4     float     sum      -1    11.77    0.00    0.00      0    11.69    0.00    0.00      0
          32             8     float     sum      -1    12.86    0.00    0.00      0    11.59    0.00    0.00      0
          64            16     float     sum      -1    11.61    0.01    0.01      0    12.21    0.01    0.01      0
         128            32     float     sum      -1    11.69    0.01    0.01      0    11.58    0.01    0.01      0
         256            64     float     sum      -1    11.78    0.02    0.02      0    12.09    0.02    0.02      0
         512           128     float     sum      -1    11.77    0.04    0.04      0    11.57    0.04    0.04      0
        1024           256     float     sum      -1    12.10    0.08    0.08      0    11.64    0.09    0.09      0
        2048           512     float     sum      -1    11.83    0.17    0.17      0    12.46    0.16    0.16      0
        4096          1024     float     sum      -1    11.79    0.35    0.35      0    11.89    0.34    0.34      0
        8192          2048     float     sum      -1    13.07    0.63    0.63      0    12.96    0.63    0.63      0
       16384          4096     float     sum      -1    17.11    0.96    0.96      0    17.07    0.96    0.96      0
       32768          8192     float     sum      -1    24.62    1.33    1.33      0    23.82    1.38    1.38      0
       65536         16384     float     sum      -1    40.10    1.63    1.63      0    39.88    1.64    1.64      0
      131072         32768     float     sum      -1    56.90    2.30    2.30      0    56.39    2.32    2.32      0
      262144         65536     float     sum      -1    86.04    3.05    3.05      0    86.83    3.02    3.02      0
      524288        131072     float     sum      -1    150.2    3.49    3.49      0    150.0    3.49    3.49      0
     1048576        262144     float     sum      -1    279.5    3.75    3.75      0    280.2    3.74    3.74      0
     2097152        524288     float     sum      -1    539.6    3.89    3.89      0    540.8    3.88    3.88      0
     4194304       1048576     float     sum      -1   1065.1    3.94    3.94      0   1062.2    3.95    3.95      0
     8388608       2097152     float     sum      -1   2113.2    3.97    3.97      0   2111.5    3.97    3.97      0
    16777216       4194304     float     sum      -1   4209.7    3.99    3.99      0   4208.0    3.99    3.99      0
    33554432       8388608     float     sum      -1   8399.7    3.99    3.99      0   8398.7    4.00    4.00      0
    67108864      16777216     float     sum      -1    16801    3.99    3.99      0    16818    3.99    3.99      0
   134217728      33554432     float     sum      -1    33587    4.00    4.00      0    33591    4.00    4.00      0
   268435456      67108864     float     sum      -1    67137    4.00    4.00      0    67169    4.00    4.00      0
   536870912     134217728     float     sum      -1   134330    4.00    4.00      0   134072    4.00    4.00      0
  1073741824     268435456     float     sum      -1   268588    4.00    4.00      0   268461    4.00    4.00      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 2.05774
#

root@autodl-container-xxxxx:~/nccl-tests# ./build/all_gather_perf -b 8 -e 1024M -f 2 -g2
# nThread 1 nGpus 2 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31745 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#  Rank  1 Group  0 Pid  31745 on autodl-container-xxxxx device  1 [0000:b5:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0     float    none      -1     0.33    0.00    0.00      0     0.36    0.00    0.00      0
           0             0     float    none      -1     0.33    0.00    0.00      0     0.33    0.00    0.00      0
          32             4     float    none      -1    10.52    0.00    0.00      0    10.01    0.00    0.00      0
          64             8     float    none      -1    10.01    0.01    0.00      0    10.07    0.01    0.00      0
         128            16     float    none      -1     9.98    0.01    0.01      0     9.99    0.01    0.01      0
         256            32     float    none      -1     9.90    0.03    0.01      0    10.15    0.03    0.01      0
         512            64     float    none      -1     9.88    0.05    0.03      0     9.99    0.05    0.03      0
        1024           128     float    none      -1    10.11    0.10    0.05      0    10.31    0.10    0.05      0
        2048           256     float    none      -1    10.02    0.20    0.10      0     9.99    0.20    0.10      0
        4096           512     float    none      -1    10.13    0.40    0.20      0     9.97    0.41    0.21      0
        8192          1024     float    none      -1    10.22    0.80    0.40      0    10.48    0.78    0.39      0
       16384          2048     float    none      -1    12.77    1.28    0.64      0    12.54    1.31    0.65      0
       32768          4096     float    none      -1    16.69    1.96    0.98      0    16.29    2.01    1.01      0
       65536          8192     float    none      -1    27.63    2.37    1.19      0    27.19    2.41    1.20      0
      131072         16384     float    none      -1    38.17    3.43    1.72      0    38.12    3.44    1.72      0
      262144         32768     float    none      -1    58.01    4.52    2.26      0    57.88    4.53    2.26      0
      524288         65536     float    none      -1    99.45    5.27    2.64      0    98.49    5.32    2.66      0
     1048576        131072     float    none      -1    181.1    5.79    2.89      0    179.1    5.85    2.93      0
     2097152        262144     float    none      -1    338.9    6.19    3.09      0    340.9    6.15    3.08      0
     4194304        524288     float    none      -1    667.0    6.29    3.14      0    671.5    6.25    3.12      0
     8388608       1048576     float    none      -1   1327.1    6.32    3.16      0   1330.5    6.30    3.15      0
    16777216       2097152     float    none      -1   2637.3    6.36    3.18      0   2639.1    6.36    3.18      0
    33554432       4194304     float    none      -1   5250.2    6.39    3.20      0   5213.9    6.44    3.22      0
    67108864       8388608     float    none      -1    10390    6.46    3.23      0   9919.0    6.77    3.38      0
   134217728      16777216     float    none      -1    20235    6.63    3.32      0    19164    7.00    3.50      0
   268435456      33554432     float    none      -1    40096    6.69    3.35      0    35687    7.52    3.76      0
   536870912      67108864     float    none      -1    79115    6.79    3.39      0    70638    7.60    3.80      0
  1073741824     134217728     float    none      -1   141614    7.58    3.79      0   139719    7.69    3.84      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 1.6651
#

root@autodl-container-xxxxx:~/nccl-tests# ./build/broadcast_perf -b 8 -e 1024M -f 2 -g2
# nThread 1 nGpus 2 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31771 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#  Rank  1 Group  0 Pid  31771 on autodl-container-xxxxx device  1 [0000:b5:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float    none       0    11.65    0.00    0.00      0    11.37    0.00    0.00      0
          16             4     float    none       0    11.47    0.00    0.00      0    11.36    0.00    0.00      0
          32             8     float    none       0    11.43    0.00    0.00      0    11.34    0.00    0.00      0
          64            16     float    none       0    11.30    0.01    0.01      0    11.59    0.01    0.01      0
         128            32     float    none       0    11.66    0.01    0.01      0    11.32    0.01    0.01      0
         256            64     float    none       0    11.49    0.02    0.02      0    11.33    0.02    0.02      0
         512           128     float    none       0    11.36    0.05    0.05      0    11.39    0.04    0.04      0
        1024           256     float    none       0    12.04    0.09    0.09      0    11.56    0.09    0.09      0
        2048           512     float    none       0    11.48    0.18    0.18      0    11.50    0.18    0.18      0
        4096          1024     float    none       0    11.52    0.36    0.36      0    11.36    0.36    0.36      0
        8192          2048     float    none       0    11.73    0.70    0.70      0    11.89    0.69    0.69      0
       16384          4096     float    none       0    12.42    1.32    1.32      0    12.41    1.32    1.32      0
       32768          8192     float    none       0    17.55    1.87    1.87      0    17.03    1.92    1.92      0
       65536         16384     float    none       0    20.73    3.16    3.16      0    19.90    3.29    3.29      0
      131072         32768     float    none       0    28.00    4.68    4.68      0    27.53    4.76    4.76      0
      262144         65536     float    none       0    39.84    6.58    6.58      0    39.75    6.59    6.59      0
      524288        131072     float    none       0    64.98    8.07    8.07      0    64.55    8.12    8.12      0
     1048576        262144     float    none       0    113.8    9.22    9.22      0    288.1    3.64    3.64      0
     2097152        524288     float    none       0    208.3   10.07   10.07      0    210.0    9.99    9.99      0
     4194304       1048576     float    none       0    400.5   10.47   10.47      0    400.3   10.48   10.48      0
     8388608       2097152     float    none       0    780.0   10.75   10.75      0    781.9   10.73   10.73      0
    16777216       4194304     float    none       0   1539.1   10.90   10.90      0   1543.7   10.87   10.87      0
    33554432       8388608     float    none       0   3054.2   10.99   10.99      0   3055.6   10.98   10.98      0
    67108864      16777216     float    none       0   6082.6   11.03   11.03      0   6079.7   11.04   11.04      0
   134217728      33554432     float    none       0    12130   11.06   11.06      0    12141   11.06   11.06      0
   268435456      67108864     float    none       0    24225   11.08   11.08      0    24229   11.08   11.08      0
   536870912     134217728     float    none       0    48458   11.08   11.08      0    48416   11.09   11.09      0
  1073741824     268435456     float    none       0    96800   11.09   11.09      0    96803   11.09   11.09      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 5.07658
#


root@autodl-container-xxxxx:~/nccl-tests# ./build/reduce_perf -b 8 -e 1024M -f 2 -g2
# nThread 1 nGpus 2 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31795 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#  Rank  1 Group  0 Pid  31795 on autodl-container-xxxxx device  1 [0000:b5:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum       0    10.39    0.00    0.00      0     9.88    0.00    0.00      0
          16             4     float     sum       0     9.83    0.00    0.00      0     9.76    0.00    0.00      0
          32             8     float     sum       0     9.86    0.00    0.00      0     9.87    0.00    0.00      0
          64            16     float     sum       0     9.82    0.01    0.01      0    10.04    0.01    0.01      0
         128            32     float     sum       0     9.92    0.01    0.01      0     9.88    0.01    0.01      0
         256            64     float     sum       0    10.27    0.02    0.02      0     9.77    0.03    0.03      0
         512           128     float     sum       0     9.87    0.05    0.05      0     9.85    0.05    0.05      0
        1024           256     float     sum       0    10.06    0.10    0.10      0     9.91    0.10    0.10      0
        2048           512     float     sum       0     9.94    0.21    0.21      0    10.12    0.20    0.20      0
        4096          1024     float     sum       0    10.03    0.41    0.41      0    10.01    0.41    0.41      0
        8192          2048     float     sum       0    10.21    0.80    0.80      0    10.26    0.80    0.80      0
       16384          4096     float     sum       0    11.76    1.39    1.39      0    11.38    1.44    1.44      0
       32768          8192     float     sum       0    16.15    2.03    2.03      0    15.99    2.05    2.05      0
       65536         16384     float     sum       0    19.10    3.43    3.43      0    19.47    3.37    3.37      0
      131072         32768     float     sum       0    26.42    4.96    4.96      0    26.39    4.97    4.97      0
      262144         65536     float     sum       0    38.99    6.72    6.72      0    39.04    6.72    6.72      0
      524288        131072     float     sum       0    63.88    8.21    8.21      0    63.64    8.24    8.24      0
     1048576        262144     float     sum       0    112.5    9.32    9.32      0    112.1    9.35    9.35      0
     2097152        524288     float     sum       0    208.9   10.04   10.04      0    208.9   10.04   10.04      0
     4194304       1048576     float     sum       0    400.1   10.48   10.48      0    398.3   10.53   10.53      0
     8388608       2097152     float     sum       0    779.9   10.76   10.76      0    778.9   10.77   10.77      0
    16777216       4194304     float     sum       0   1535.6   10.93   10.93      0   1535.8   10.92   10.92      0
    33554432       8388608     float     sum       0   3045.9   11.02   11.02      0   3044.4   11.02   11.02      0
    67108864      16777216     float     sum       0   6062.1   11.07   11.07      0   6058.6   11.08   11.08      0
   134217728      33554432     float     sum       0    12094   11.10   11.10      0    12094   11.10   11.10      0
   268435456      67108864     float     sum       0    24162   11.11   11.11      0    24154   11.11   11.11      0
   536870912     134217728     float     sum       0    48281   11.12   11.12      0    48234   11.13   11.13      0
  1073741824     268435456     float     sum       0    96427   11.14   11.14      0    96450   11.13   11.13      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 5.23248
#

root@autodl-container-xxxxx:~/nccl-tests# ./build/reduce_perf -b 8 -e 1024M -f 2
# nThread 1 nGpus 1 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31816 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw     busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)    (GB/s) 
           8             2     float     sum       0     4.21    0.00    0.00      0     0.19    0.04       0.04        0
          16             4     float     sum       0     4.00    0.00    0.00      0     0.19    0.08       0.08        0
          32             8     float     sum       0     4.27    0.01    0.01      0     0.19    0.17       0.17        0
          64            16     float     sum       0     4.25    0.02    0.02      0     0.19    0.34       0.34        0
         128            32     float     sum       0     3.99    0.03    0.03      0     0.18    0.69       0.69        0
         256            64     float     sum       0     4.25    0.06    0.06      0     0.19    1.36       1.36        0
         512           128     float     sum       0     4.34    0.12    0.12      0     0.19    2.76       2.76        0
        1024           256     float     sum       0     4.14    0.25    0.25      0     0.19    5.48       5.48        0
        2048           512     float     sum       0     4.17    0.49    0.49      0     0.18   11.17       11.17       0
        4096          1024     float     sum       0     4.24    0.97    0.97      0     0.19   21.83       21.83       0
        8192          2048     float     sum       0     4.13    1.98    1.98      0     0.19   44.03       44.03       0
       16384          4096     float     sum       0     4.11    3.99    3.99      0     0.19   87.66       87.66       0
       32768          8192     float     sum       0     4.11    7.96    7.96      0     0.19   176.31      176.31      0
       65536         16384     float     sum       0     4.18   15.67   15.67      0     0.19   353.20      353.20      0
      131072         32768     float     sum       0     4.15   31.61   31.61      0     0.19   706.02      706.02      0
      262144         65536     float     sum       0     4.16   62.97   62.97      0     0.19   1356.15     1356.15     0
      524288        131072     float     sum       0     4.48  116.92  116.92      0     0.19   2796.95     2796.95     0
     1048576        262144     float     sum       0     6.42  163.33  163.33      0     0.19   5601.37     5601.37     0
     2097152        524288     float     sum       0    10.12  207.23  207.23      0     0.19   11281.08    11281.08    0
     4194304       1048576     float     sum       0    18.26  229.73  229.73      0     0.19   22162.77    22162.77    0
     8388608       2097152     float     sum       0    33.70  248.94  248.94      0     0.19   44667.77    44667.77    0
    16777216       4194304     float     sum       0    64.46  260.29  260.29      0     0.19   88370.90    88370.90    0
    33554432       8388608     float     sum       0    127.7  262.84  262.84      0     0.19   180303.23   180303.23   0
    67108864      16777216     float     sum       0    253.1  265.13  265.13      0     0.19   358200.50   358200.50   0
   134217728      33554432     float     sum       0    504.0  266.33  266.33      0     0.19   712219.30   712219.30   0
   268435456      67108864     float     sum       0   1005.5  266.97  266.97      0     0.19   1397737.34  1397737.34  0
   536870912     134217728     float     sum       0   2010.4  267.04  267.04      0     0.19   2821923.32  2821923.32  0
  1073741824     268435456     float     sum       0   4016.0  267.37  267.37      0     0.19   5706839.35  5706839.35  0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 202818
#


root@autodl-container-xxxxx:~/nccl-tests# ./build/alltoall_perf -b 8 -e 1024M -f 2 -g 2
# nThread 1 nGpus 2 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  31856 on autodl-container-xxxxx device  0 [0000:b4:00] NVIDIA GeForce RTX 2080 Ti
#  Rank  1 Group  0 Pid  31856 on autodl-container-xxxxx device  1 [0000:b5:00] NVIDIA GeForce RTX 2080 Ti
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0     float    none      -1    12.97    0.00    0.00      0    12.86    0.00    0.00    N/A
           0             0     float    none      -1    12.99    0.00    0.00      0    12.85    0.00    0.00    N/A
          32             4     float    none      -1    13.51    0.00    0.00      0    13.11    0.00    0.00    N/A
          64             8     float    none      -1    13.07    0.00    0.00      0    12.90    0.00    0.00    N/A
         128            16     float    none      -1    13.07    0.01    0.00      0    13.11    0.01    0.00    N/A
         256            32     float    none      -1    12.97    0.02    0.01      0    12.84    0.02    0.01    N/A
         512            64     float    none      -1    12.86    0.04    0.02      0    12.85    0.04    0.02    N/A
        1024           128     float    none      -1    12.94    0.08    0.04      0    12.89    0.08    0.04    N/A
        2048           256     float    none      -1    12.87    0.16    0.08      0    13.04    0.16    0.08    N/A
        4096           512     float    none      -1    13.03    0.31    0.16      0    12.97    0.32    0.16    N/A
        8192          1024     float    none      -1    13.18    0.62    0.31      0    13.11    0.62    0.31    N/A
       16384          2048     float    none      -1    15.03    1.09    0.54      0    14.80    1.11    0.55    N/A
       32768          4096     float    none      -1    20.22    1.62    0.81      0    19.81    1.65    0.83    N/A
       65536          8192     float    none      -1    24.15    2.71    1.36      0    23.76    2.76    1.38    N/A
      131072         16384     float    none      -1    32.18    4.07    2.04      0    32.92    3.98    1.99    N/A
      262144         32768     float    none      -1    53.51    4.90    2.45      0    47.45    5.52    2.76    N/A
      524288         65536     float    none      -1    87.24    6.01    3.00      0    87.55    5.99    2.99    N/A
     1048576        131072     float    none      -1    151.6    6.92    3.46      0    150.7    6.96    3.48    N/A
     2097152        262144     float    none      -1    274.3    7.65    3.82      0    282.1    7.43    3.72    N/A
     4194304        524288     float    none      -1    541.5    7.75    3.87      0    538.3    7.79    3.90    N/A
     8388608       1048576     float    none      -1   1060.7    7.91    3.95      0   1061.7    7.90    3.95    N/A
    16777216       2097152     float    none      -1   1886.6    8.89    4.45      0   1870.3    8.97    4.49    N/A
    33554432       4194304     float    none      -1   3743.2    8.96    4.48      0   3781.3    8.87    4.44    N/A
    67108864       8388608     float    none      -1   7365.2    9.11    4.56      0   7517.4    8.93    4.46    N/A
   134217728      16777216     float    none      -1    14735    9.11    4.55      0    14769    9.09    4.54    N/A
   268435456      33554432     float    none      -1    29418    9.12    4.56      0    29062    9.24    4.62    N/A
   536870912      67108864     float    none      -1    58597    9.16    4.58      0    56988    9.42    4.71    N/A
  1073741824     134217728     float    none      -1   116573    9.21    4.61      0   113591    9.45    4.73    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 2.06935
#
```