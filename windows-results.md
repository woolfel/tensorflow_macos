# Summary
For comparison, I ran similar test on windows for a point of reference.

## Hardware
* AMD Ryzen 3700x
* RTX 2060 6G GDDR6
* 64G DDR4 3600
* Crucial P2 1TB 3D NAND NVMe PCIe M.2 SSD Up to 2400 MB/s

## Software
* Python 3.9.6
* Tensorflow 2.5.0 with GPU
* CUDA 11.2

<pre>
2021-11-02 20:43:22.950473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-02 20:43:22.953029: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-02 20:43:22.955406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-02 20:43:22.955524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-02 20:43:24.027164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-02 20:43:24.027264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0
2021-11-02 20:43:24.028138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N
2021-11-02 20:43:24.030790: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3961 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:0a:00.0, compute capability: 7.5)
Dataset mnist downloaded and prepared to C:\Users\peter\tensorflow_datasets\mnist\3.0.1. Subsequent calls will reuse this data.
Epoch 1/12
2021-11-02 20:43:38.514411: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-02 20:43:39.284489: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-02 20:43:40.529060: I tensorflow/stream_executor/cuda/cuda_dnn.cc:359] Loaded cuDNN version 8101
2021-11-02 20:43:42.407152: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-02 20:43:43.664602: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
469/469 [==============================] - 10s 8ms/step - loss: 0.1576 - accuracy: 0.9535 - val_loss: 0.0498 - val_accuracy: 0.9830
Epoch 2/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0433 - accuracy: 0.9870 - val_loss: 0.0389 - val_accuracy: 0.9880
Epoch 3/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0278 - accuracy: 0.9915 - val_loss: 0.0343 - val_accuracy: 0.9889
Epoch 4/12
469/469 [==============================] - 4s 7ms/step - loss: 0.0177 - accuracy: 0.9946 - val_loss: 0.0358 - val_accuracy: 0.9896
Epoch 5/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0128 - accuracy: 0.9955 - val_loss: 0.0354 - val_accuracy: 0.9895
Epoch 6/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0099 - accuracy: 0.9968 - val_loss: 0.0390 - val_accuracy: 0.9898
Epoch 7/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0077 - accuracy: 0.9973 - val_loss: 0.0398 - val_accuracy: 0.9888
Epoch 8/12
469/469 [==============================] - 4s 9ms/step - loss: 0.0065 - accuracy: 0.9979 - val_loss: 0.0398 - val_accuracy: 0.9894
Epoch 9/12
469/469 [==============================] - 4s 8ms/step - loss: 0.0054 - accuracy: 0.9983 - val_loss: 0.0537 - val_accuracy: 0.9876
Epoch 10/12
469/469 [==============================] - 4s 7ms/step - loss: 0.0056 - accuracy: 0.9982 - val_loss: 0.0406 - val_accuracy: 0.9897
Epoch 11/12
469/469 [==============================] - 4s 7ms/step - loss: 0.0045 - accuracy: 0.9986 - val_loss: 0.0349 - val_accuracy: 0.9910
Epoch 12/12
469/469 [==============================] - 3s 7ms/step - loss: 0.0027 - accuracy: 0.9991 - val_loss: 0.0485 - val_accuracy: 0.9890
</pre>

## Memory limit of RTX 2060 6G

I increased the batch size until tensorflow started to give me memory warnings. At batch size 4096, tensorflow gives me warnings, but it still runs.

<pre>
PS G:\fashionmnist> python .\tf_fmnist_benchmark.py
2021-11-05 09:30:43.047596: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2.5.0
2021-11-05 09:30:45.010298: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library nvcuda.dll
2021-11-05 09:30:45.063473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-05 09:30:45.063595: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2021-11-05 09:30:45.075178: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-05 09:30:45.075251: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2021-11-05 09:30:45.078338: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cufft64_10.dll
2021-11-05 09:30:45.079255: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library curand64_10.dll
2021-11-05 09:30:45.083224: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusolver64_11.dll
2021-11-05 09:30:45.085660: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusparse64_11.dll
2021-11-05 09:30:45.086159: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-05 09:30:45.086458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-05 09:30:45.087059: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-05 09:30:45.090456: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-05 09:30:45.090573: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-05 09:30:45.534974: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-05 09:30:45.535077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0
2021-11-05 09:30:45.535904: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N
2021-11-05 09:30:45.536640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3961 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:0a:00.0, compute capability: 7.5)
Epoch 1/15
2021-11-05 09:30:45.965141: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-05 09:30:46.824558: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-05 09:30:47.229584: I tensorflow/stream_executor/cuda/cuda_dnn.cc:359] Loaded cuDNN version 8101
2021-11-05 09:30:47.824028: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-05 09:30:48.240581: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2021-11-05 09:30:48.424570: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:48.424739: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:48.882287: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:48.882453: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:48.961838: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:48.961997: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:49.124881: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.77GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-05 09:30:49.125041: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.77GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
15/15 [==============================] - 7s 243ms/step - loss: 1.1497 - accuracy: 0.6212 - val_loss: 0.6793 - val_accuracy: 0.7526
Epoch 2/15
15/15 [==============================] - 2s 122ms/step - loss: 0.5870 - accuracy: 0.7871 - val_loss: 0.5394 - val_accuracy: 0.8050
Epoch 3/15
15/15 [==============================] - 2s 122ms/step - loss: 0.4839 - accuracy: 0.8295 - val_loss: 0.4683 - val_accuracy: 0.8322
Epoch 4/15
15/15 [==============================] - 2s 122ms/step - loss: 0.4281 - accuracy: 0.8492 - val_loss: 0.4374 - val_accuracy: 0.8415
Epoch 5/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3950 - accuracy: 0.8611 - val_loss: 0.4118 - val_accuracy: 0.8519
Epoch 6/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3722 - accuracy: 0.8697 - val_loss: 0.3932 - val_accuracy: 0.8612
Epoch 7/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3550 - accuracy: 0.8752 - val_loss: 0.3713 - val_accuracy: 0.8710
Epoch 8/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3345 - accuracy: 0.8828 - val_loss: 0.3597 - val_accuracy: 0.8735
Epoch 9/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3243 - accuracy: 0.8855 - val_loss: 0.3534 - val_accuracy: 0.8751
Epoch 10/15
15/15 [==============================] - 2s 122ms/step - loss: 0.3090 - accuracy: 0.8911 - val_loss: 0.3285 - val_accuracy: 0.8825
Epoch 11/15
15/15 [==============================] - 2s 122ms/step - loss: 0.2982 - accuracy: 0.8943 - val_loss: 0.3208 - val_accuracy: 0.8857
Epoch 12/15
15/15 [==============================] - 2s 122ms/step - loss: 0.2832 - accuracy: 0.9007 - val_loss: 0.3130 - val_accuracy: 0.8895
Epoch 13/15
15/15 [==============================] - 2s 122ms/step - loss: 0.2728 - accuracy: 0.9036 - val_loss: 0.3041 - val_accuracy: 0.8921
Epoch 14/15
15/15 [==============================] - 2s 122ms/step - loss: 0.2625 - accuracy: 0.9082 - val_loss: 0.2962 - val_accuracy: 0.8935
Epoch 15/15
15/15 [==============================] - 2s 122ms/step - loss: 0.2523 - accuracy: 0.9113 - val_loss: 0.2926 - val_accuracy: 0.8943
</pre>

## Impact of memory on Training

With a RTX 2060 6G card, the practical limit is 5.5G. The way CUDA support works in tensorflow, it allocates as much memory as possible at startup. There's no practical way to see exactly how much memory tensorflow uses from windows. There's NVidia tools to inspect memory usage on the video card.

To figure out the practical limit, we can look at fashion mnist dataset. After tensorflow_dataset converts it to tfrecords, the size of the files:

* test 5499 KB or 5.37 MB
* train 32889 KB or 32.12 MB

We know with batch 4096 it gets memory errors. From the MacOS benchmark, we see tensorflow set to batch 256 uses 1.3G of memory. Which would mean the ratio of dataset size to memory used:

* train + test = 37.5 MB
* 35 = 1300 / 37.5

If we look at COCO dataset after tensorflow_datasets splits the data, each file is roughly 105 MB. The actual file size varies from 104 to 108 MB, but I'll use 100 MB to make things easy. For each run TF will load train and test data, which is roughly 200 MB.

* 200 x 35 = 7G projected memory requirement

When the memory requirements exceeds physical memory, the system will use swap. In many cases, it might still run, but it will take longer to train.
