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
