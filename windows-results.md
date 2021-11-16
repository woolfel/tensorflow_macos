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

## fashion mnist

<pre>
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

# the benchmark loads the MNIST dataset from tensorflow datasets
# a possible alternative is fashion MNIST, which should require more power
(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# you can change the batch size to see how it performs. Larger batch size will stress GPU more
batch_size = 256

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.Conv2D(32, kernel_size=(1, 1),
                 activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#   tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)
</pre>

<pre>
PS G:\fashionmnist> python .\tf_fmnist_benchmark.py
2021-11-06 19:06:44.557525: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2.5.0
2021-11-06 19:06:48.615675: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library nvcuda.dll
2021-11-06 19:06:48.647442: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-06 19:06:48.647585: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2021-11-06 19:06:49.048809: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-06 19:06:49.048928: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2021-11-06 19:06:49.302037: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cufft64_10.dll
2021-11-06 19:06:49.324907: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library curand64_10.dll
2021-11-06 19:06:49.522874: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusolver64_11.dll
2021-11-06 19:06:49.714937: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusparse64_11.dll
2021-11-06 19:06:49.730149: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-06 19:06:49.730302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-06 19:06:49.730980: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 19:06:49.731883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-06 19:06:49.732189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-06 19:06:50.199543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 19:06:50.199655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0
2021-11-06 19:06:50.200651: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N
2021-11-06 19:06:50.201265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3961 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:0a:00.0, compute capability: 7.5)
Epoch 1/15
2021-11-06 19:06:50.715509: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-06 19:06:51.656465: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-06 19:06:52.269722: I tensorflow/stream_executor/cuda/cuda_dnn.cc:359] Loaded cuDNN version 8101
2021-11-06 19:06:53.131559: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-06 19:06:53.538432: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
235/235 [==============================] - 9s 21ms/step - loss: 0.5293 - accuracy: 0.8099 - val_loss: 0.3679 - val_accuracy: 0.8668
Epoch 2/15
235/235 [==============================] - 5s 19ms/step - loss: 0.3062 - accuracy: 0.8907 - val_loss: 0.2990 - val_accuracy: 0.8917
Epoch 3/15
235/235 [==============================] - 5s 19ms/step - loss: 0.2527 - accuracy: 0.9083 - val_loss: 0.2717 - val_accuracy: 0.9022
Epoch 4/15
235/235 [==============================] - 5s 19ms/step - loss: 0.2251 - accuracy: 0.9175 - val_loss: 0.2446 - val_accuracy: 0.9110
Epoch 5/15
235/235 [==============================] - 5s 19ms/step - loss: 0.1959 - accuracy: 0.9274 - val_loss: 0.2461 - val_accuracy: 0.9149
Epoch 6/15
235/235 [==============================] - 5s 19ms/step - loss: 0.1782 - accuracy: 0.9347 - val_loss: 0.2449 - val_accuracy: 0.9139
Epoch 7/15
235/235 [==============================] - 5s 19ms/step - loss: 0.1547 - accuracy: 0.9423 - val_loss: 0.2385 - val_accuracy: 0.9160
Epoch 8/15
235/235 [==============================] - 5s 19ms/step - loss: 0.1398 - accuracy: 0.9488 - val_loss: 0.2530 - val_accuracy: 0.9164
Epoch 9/15
235/235 [==============================] - 5s 19ms/step - loss: 0.1217 - accuracy: 0.9549 - val_loss: 0.2411 - val_accuracy: 0.9217
Epoch 10/15
235/235 [==============================] - 5s 20ms/step - loss: 0.1027 - accuracy: 0.9613 - val_loss: 0.2626 - val_accuracy: 0.9216
Epoch 11/15
235/235 [==============================] - 5s 19ms/step - loss: 0.0893 - accuracy: 0.9662 - val_loss: 0.2702 - val_accuracy: 0.9209
Epoch 12/15
235/235 [==============================] - 5s 19ms/step - loss: 0.0771 - accuracy: 0.9721 - val_loss: 0.3021 - val_accuracy: 0.9201
Epoch 13/15
235/235 [==============================] - 5s 19ms/step - loss: 0.0669 - accuracy: 0.9750 - val_loss: 0.3301 - val_accuracy: 0.9152
Epoch 14/15
235/235 [==============================] - 5s 20ms/step - loss: 0.0492 - accuracy: 0.9817 - val_loss: 0.3672 - val_accuracy: 0.9116
Epoch 15/15
235/235 [==============================] - 5s 19ms/step - loss: 0.0462 - accuracy: 0.9834 - val_loss: 0.4054 - val_accuracy: 0.9170
</pre>

## Memory limit of RTX 2060 6G

I increased the batch size until tensorflow started to give me memory warnings. At batch size 4096, tensorflow gives me warnings, but it still runs.

<pre>
PS G:\fashionmnist> python .\tf_fmnist_benchmark.py
2021-11-07 08:09:48.982672: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2.5.0
2021-11-07 08:09:50.928591: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library nvcuda.dll
2021-11-07 08:09:50.956912: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-07 08:09:50.957020: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2021-11-07 08:09:50.968094: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-07 08:09:50.968163: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2021-11-07 08:09:50.971264: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cufft64_10.dll
2021-11-07 08:09:50.972197: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library curand64_10.dll
2021-11-07 08:09:50.976312: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusolver64_11.dll
2021-11-07 08:09:50.978768: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusparse64_11.dll
2021-11-07 08:09:50.979184: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-07 08:09:50.979297: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-07 08:09:50.979872: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-07 08:09:50.982227: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:0a:00.0 name: GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2021-11-07 08:09:50.982897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-11-07 08:09:51.407728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-07 08:09:51.407847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0
2021-11-07 08:09:51.408895: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N
2021-11-07 08:09:51.409113: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3961 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:0a:00.0, compute capability: 7.5)
Epoch 1/15
2021-11-07 08:09:51.884153: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-07 08:09:52.672148: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2021-11-07 08:09:53.080443: I tensorflow/stream_executor/cuda/cuda_dnn.cc:359] Loaded cuDNN version 8101
2021-11-07 08:09:53.646935: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2021-11-07 08:09:54.043437: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2021-11-07 08:09:54.258990: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.259152: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.710250: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.17GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.710399: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.17GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.766104: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.766255: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.767023: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.767443: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.863704: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-07 08:09:54.863855: W tensorflow/core/common_runtime/bfc_allocator.cc:271] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
15/15 [==============================] - 12s 483ms/step - loss: 1.5502 - accuracy: 0.4406 - val_loss: 0.9833 - val_accuracy: 0.6618
Epoch 2/15
15/15 [==============================] - 4s 258ms/step - loss: 0.8061 - accuracy: 0.7056 - val_loss: 0.6691 - val_accuracy: 0.7483
Epoch 3/15
15/15 [==============================] - 4s 258ms/step - loss: 0.5819 - accuracy: 0.7807 - val_loss: 0.5340 - val_accuracy: 0.8044
Epoch 4/15
15/15 [==============================] - 4s 259ms/step - loss: 0.4780 - accuracy: 0.8303 - val_loss: 0.4620 - val_accuracy: 0.8355
Epoch 5/15
15/15 [==============================] - 4s 258ms/step - loss: 0.4194 - accuracy: 0.8515 - val_loss: 0.4241 - val_accuracy: 0.8457
Epoch 6/15
15/15 [==============================] - 4s 258ms/step - loss: 0.3805 - accuracy: 0.8650 - val_loss: 0.3905 - val_accuracy: 0.8598
Epoch 7/15
15/15 [==============================] - 4s 258ms/step - loss: 0.3614 - accuracy: 0.8716 - val_loss: 0.3878 - val_accuracy: 0.8582
Epoch 8/15
15/15 [==============================] - 4s 259ms/step - loss: 0.3441 - accuracy: 0.8764 - val_loss: 0.3562 - val_accuracy: 0.8728
Epoch 9/15
15/15 [==============================] - 4s 258ms/step - loss: 0.3219 - accuracy: 0.8844 - val_loss: 0.3390 - val_accuracy: 0.8792
Epoch 10/15
15/15 [==============================] - 4s 259ms/step - loss: 0.3061 - accuracy: 0.8906 - val_loss: 0.3355 - val_accuracy: 0.8788
Epoch 11/15
15/15 [==============================] - 4s 259ms/step - loss: 0.2920 - accuracy: 0.8957 - val_loss: 0.3162 - val_accuracy: 0.8873
Epoch 12/15
15/15 [==============================] - 4s 260ms/step - loss: 0.2766 - accuracy: 0.9020 - val_loss: 0.3037 - val_accuracy: 0.8904
Epoch 13/15
15/15 [==============================] - 4s 260ms/step - loss: 0.2683 - accuracy: 0.9044 - val_loss: 0.2971 - val_accuracy: 0.8946
Epoch 14/15
15/15 [==============================] - 4s 260ms/step - loss: 0.2617 - accuracy: 0.9061 - val_loss: 0.2943 - val_accuracy: 0.8947
Epoch 15/15
15/15 [==============================] - 4s 260ms/step - loss: 0.2474 - accuracy: 0.9112 - val_loss: 0.2941 - val_accuracy: 0.8919
</pre>

## Impact of memory on Training

With a RTX 2060 6G card, the practical limit is 5.5G. The way CUDA support works in tensorflow, it allocates as much memory as possible at startup. There's no practical way to see exactly how much memory tensorflow uses from windows. There's NVidia tools to inspect memory usage on the video card.

To figure out the practical limit, we can look at fashion mnist dataset. After tensorflow_dataset converts it to tfrecords, the size of the files:

* test 5499 KB or 5.37 MB
* train 32889 KB or 32.12 MB

We know with batch 4096 it gets memory errors. From the MacOS benchmark results, we see tensorflow set to batch 256 uses 1.3G of memory. Which would mean the ratio of dataset size to memory used:

* train + test = 37.5 MB
* 35 = 1300 / 37.5

It's important to keep in mind the memory used varies with batch size. If you're using batch 256, we can do some simple calculation to estimate. If we look at COCO dataset after tensorflow_datasets splits the data, each file is roughly 105 MB. The actual file size varies from 104 to 108 MB, but I'll use 100 MB to make things easy. For each run TF will load train and test data, which is roughly 200 MB.

* 200 x 35 = 7G projected memory requirement

When the memory requirements exceeds physical memory, the system will use swap. In many cases, it might still run, but it will take longer to train. If you want to learn more about batch size recommendations, there are plenty of YouTube videos that give a good overview. For small models with small datasets, 32 to 128 batch size is usually sufficient. If you're using large datasets with more than 100 million records, you'll likely want to use much larger batch size to speed up the training time. In those situations, you really want to train on the cloud.
