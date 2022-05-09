# Hardware

## M1 Macbook Air
* 16G memory
* Monterey
* Python 3 ARM64

## M1 Max Macbook Pro
* 32G memory
* 24 GPU
* Monterey
* Python 3 ARM64

## Training settings

* Batch Size: 128
* GPU Average: 85%

<pre>
peter@AirPete bin % time ./python3 /Users/peter/tf-train-test.py
2.4.0-rc0
2021-10-27 08:40:14.980015: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-10-27 08:40:14.980492: W tensorflow/core/platform/profile_utils/cpu_utils.cc:126] Failed to get CPU frequency: 0 Hz
2021-10-27 08:40:15.215032: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Train on 469 steps, validate on 79 steps
Epoch 1/12
469/469 [==============================] - ETA: 0s - batch: 234.0000 - size: 1.0000 - loss: 0.1576 - accuracy: 0.9530/Users/peter/tensorflow_macos_venv/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:2325: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  warnings.warn('`Model.state_updates` will be removed in a future version. '
469/469 [==============================] - 13s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.1576 - accuracy: 0.9530 - val_loss: 0.0537 - val_accuracy: 0.9832
Epoch 2/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0461 - accuracy: 0.9856 - val_loss: 0.0383 - val_accuracy: 0.9891
Epoch 3/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0292 - accuracy: 0.9905 - val_loss: 0.0315 - val_accuracy: 0.9902
Epoch 4/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0190 - accuracy: 0.9937 - val_loss: 0.0439 - val_accuracy: 0.9867
Epoch 5/12
469/469 [==============================] - 13s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0137 - accuracy: 0.9957 - val_loss: 0.0374 - val_accuracy: 0.9875
Epoch 6/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0102 - accuracy: 0.9966 - val_loss: 0.0347 - val_accuracy: 0.9907
Epoch 7/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0089 - accuracy: 0.9972 - val_loss: 0.0398 - val_accuracy: 0.9891
Epoch 8/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0075 - accuracy: 0.9975 - val_loss: 0.0383 - val_accuracy: 0.9894
Epoch 9/12
469/469 [==============================] - 13s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0062 - accuracy: 0.9979 - val_loss: 0.0381 - val_accuracy: 0.9900
Epoch 10/12
469/469 [==============================] - 13s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0051 - accuracy: 0.9982 - val_loss: 0.0461 - val_accuracy: 0.9891
Epoch 11/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0041 - accuracy: 0.9987 - val_loss: 0.0457 - val_accuracy: 0.9896
Epoch 12/12
469/469 [==============================] - 12s 25ms/step - batch: 234.0000 - size: 1.0000 - loss: 0.0041 - accuracy: 0.9987 - val_loss: 0.0499 - val_accuracy: 0.9885
./python3 /Users/peter/tf-train-test.py  97.72s user 25.94s system 80% cpu 2:33.11 total
</pre>

Batch Size: 256
GPU Average: 85-90%
<pre>
peter@AirPete bin % time ./python3 /Users/peter/tf-train-test.py 
2.4.0-rc0
2021-10-27 08:29:48.064385: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-10-27 08:29:48.064854: W tensorflow/core/platform/profile_utils/cpu_utils.cc:126] Failed to get CPU frequency: 0 Hz
2021-10-27 08:29:48.303795: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Train on 235 steps, validate on 40 steps
Epoch 1/12
234/235 [============================>.] - ETA: 0s - batch: 116.5000 - size: 1.0000 - loss: 0.2167 - accuracy: 0.9378/Users/peter/tensorflow_macos_venv/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:2325: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  warnings.warn('`Model.state_updates` will be removed in a future version. '
235/235 [==============================] - 12s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.2161 - accuracy: 0.9379 - val_loss: 0.0704 - val_accuracy: 0.9798
Epoch 2/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0547 - accuracy: 0.9838 - val_loss: 0.0468 - val_accuracy: 0.9852
Epoch 3/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0367 - accuracy: 0.9886 - val_loss: 0.0349 - val_accuracy: 0.9875
Epoch 4/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0259 - accuracy: 0.9919 - val_loss: 0.0330 - val_accuracy: 0.9888
Epoch 5/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0178 - accuracy: 0.9943 - val_loss: 0.0433 - val_accuracy: 0.9867
Epoch 6/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0139 - accuracy: 0.9958 - val_loss: 0.0358 - val_accuracy: 0.9894
Epoch 7/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0091 - accuracy: 0.9972 - val_loss: 0.0334 - val_accuracy: 0.9904
Epoch 8/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0080 - accuracy: 0.9972 - val_loss: 0.0360 - val_accuracy: 0.9896
Epoch 9/12
235/235 [==============================] - 12s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0062 - accuracy: 0.9979 - val_loss: 0.0350 - val_accuracy: 0.9900
Epoch 10/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0038 - accuracy: 0.9988 - val_loss: 0.0344 - val_accuracy: 0.9899
Epoch 11/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0044 - accuracy: 0.9984 - val_loss: 0.0392 - val_accuracy: 0.9897
Epoch 12/12
235/235 [==============================] - 11s 45ms/step - batch: 117.0000 - size: 1.0000 - loss: 0.0079 - accuracy: 0.9975 - val_loss: 0.0424 - val_accuracy: 0.9894
./python3 /Users/peter/tf-train-test.py  86.64s user 23.11s system 78% cpu 2:20.20 total
</pre>

Batch Size: 512
GPU Average: 90%
<pre>
peter@AirPete bin % time ./python3 /Users/peter/tf-train-test.py
2.4.0-rc0
2021-10-27 08:35:41.481859: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-10-27 08:35:41.482343: W tensorflow/core/platform/profile_utils/cpu_utils.cc:126] Failed to get CPU frequency: 0 Hz
2021-10-27 08:35:41.716506: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Train on 118 steps, validate on 20 steps
Epoch 1/12
118/118 [==============================] - ETA: 0s - batch: 58.5000 - size: 1.0000 - loss: 0.3117 - accuracy: 0.9087/Users/peter/tensorflow_macos_venv/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:2325: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  warnings.warn('`Model.state_updates` will be removed in a future version. '
118/118 [==============================] - 11s 87ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.3117 - accuracy: 0.9087 - val_loss: 0.0967 - val_accuracy: 0.9705
Epoch 2/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0702 - accuracy: 0.9794 - val_loss: 0.0474 - val_accuracy: 0.9838
Epoch 3/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0444 - accuracy: 0.9865 - val_loss: 0.0449 - val_accuracy: 0.9852
Epoch 4/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0342 - accuracy: 0.9894 - val_loss: 0.0334 - val_accuracy: 0.9894
Epoch 5/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0250 - accuracy: 0.9921 - val_loss: 0.0398 - val_accuracy: 0.9869
Epoch 6/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0208 - accuracy: 0.9936 - val_loss: 0.0318 - val_accuracy: 0.9895
Epoch 7/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0139 - accuracy: 0.9958 - val_loss: 0.0331 - val_accuracy: 0.9886
Epoch 8/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0113 - accuracy: 0.9966 - val_loss: 0.0294 - val_accuracy: 0.9905
Epoch 9/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0088 - accuracy: 0.9973 - val_loss: 0.0395 - val_accuracy: 0.9885
Epoch 10/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0067 - accuracy: 0.9982 - val_loss: 0.0357 - val_accuracy: 0.9906
Epoch 11/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0055 - accuracy: 0.9984 - val_loss: 0.0370 - val_accuracy: 0.9894
Epoch 12/12
118/118 [==============================] - 11s 88ms/step - batch: 58.5000 - size: 1.0000 - loss: 0.0041 - accuracy: 0.9986 - val_loss: 0.0356 - val_accuracy: 0.9899
./python3 /Users/peter/tf-train-test.py  86.64s user 23.13s system 79% cpu 2:17.89 total
</pre>

Results with tensorflow-metal and fashion_mnist

<pre>
(tensorflow_ml) peter@AirPete benchmarks % python3 ./tfmetal_mnist_test.py 
2.6.0
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2021-11-03 11:52:31.428228: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2021-11-03 11:52:31.428330: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Epoch 1/12
2021-11-03 11:52:31.664780: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-03 11:52:31.665591: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
2021-11-03 11:52:31.665641: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
469/469 [==============================] - ETA: 0s - loss: 0.4252 - accuracy: 0.84792021-11-03 11:52:40.624865: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
469/469 [==============================] - 10s 19ms/step - loss: 0.4252 - accuracy: 0.8479 - val_loss: 0.3273 - val_accuracy: 0.8838
Epoch 2/12
469/469 [==============================] - 9s 18ms/step - loss: 0.2653 - accuracy: 0.9049 - val_loss: 0.2766 - val_accuracy: 0.9012
Epoch 3/12
469/469 [==============================] - 9s 18ms/step - loss: 0.2164 - accuracy: 0.9203 - val_loss: 0.2520 - val_accuracy: 0.9083
Epoch 4/12
469/469 [==============================] - 9s 18ms/step - loss: 0.1828 - accuracy: 0.9328 - val_loss: 0.2325 - val_accuracy: 0.9177
Epoch 5/12
469/469 [==============================] - 9s 18ms/step - loss: 0.1532 - accuracy: 0.9443 - val_loss: 0.2364 - val_accuracy: 0.9186
Epoch 6/12
469/469 [==============================] - 9s 18ms/step - loss: 0.1273 - accuracy: 0.9531 - val_loss: 0.2391 - val_accuracy: 0.9182
Epoch 7/12
469/469 [==============================] - 9s 18ms/step - loss: 0.1034 - accuracy: 0.9627 - val_loss: 0.2504 - val_accuracy: 0.9200
Epoch 8/12
469/469 [==============================] - 9s 18ms/step - loss: 0.0845 - accuracy: 0.9689 - val_loss: 0.2523 - val_accuracy: 0.9205
Epoch 9/12
469/469 [==============================] - 9s 18ms/step - loss: 0.0665 - accuracy: 0.9757 - val_loss: 0.2745 - val_accuracy: 0.9199
Epoch 10/12
469/469 [==============================] - 9s 18ms/step - loss: 0.0547 - accuracy: 0.9805 - val_loss: 0.2830 - val_accuracy: 0.9217
Epoch 11/12
469/469 [==============================] - 9s 18ms/step - loss: 0.0414 - accuracy: 0.9856 - val_loss: 0.3037 - val_accuracy: 0.9209
Epoch 12/12
469/469 [==============================] - 9s 18ms/step - loss: 0.0331 - accuracy: 0.9883 - val_loss: 0.3314 - val_accuracy: 0.9200
</pre>

# M1 Max Macbook Pro

<pre>
(tensorflow_ml) peter@Peters-MBP benchmarks % python3 tfmetal_fmnist_test.py
2.6.0
Metal device set to: Apple M1 Max

systemMemory: 32.00 GB
maxCacheSize: 10.67 GB

2021-11-16 18:53:47.719829: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2021-11-16 18:53:47.719943: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Epoch 1/15
2021-11-16 18:53:48.002876: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2021-11-16 18:53:48.003886: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
2021-11-16 18:53:48.003946: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
234/235 [============================>.] - ETA: 0s - loss: 0.5547 - accuracy: 0.79982021-11-16 18:53:53.367536: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
235/235 [==============================] - 6s 20ms/step - loss: 0.5544 - accuracy: 0.7998 - val_loss: 0.3760 - val_accuracy: 0.8684
Epoch 2/15
235/235 [==============================] - 4s 18ms/step - loss: 0.3148 - accuracy: 0.8873 - val_loss: 0.3073 - val_accuracy: 0.8876
Epoch 3/15
235/235 [==============================] - 4s 19ms/step - loss: 0.2605 - accuracy: 0.9057 - val_loss: 0.2977 - val_accuracy: 0.8927
Epoch 4/15
235/235 [==============================] - 4s 18ms/step - loss: 0.2208 - accuracy: 0.9181 - val_loss: 0.2623 - val_accuracy: 0.9082
Epoch 5/15
235/235 [==============================] - 4s 19ms/step - loss: 0.1941 - accuracy: 0.9289 - val_loss: 0.2502 - val_accuracy: 0.9105
Epoch 6/15
235/235 [==============================] - 4s 18ms/step - loss: 0.1751 - accuracy: 0.9347 - val_loss: 0.2470 - val_accuracy: 0.9131
Epoch 7/15
235/235 [==============================] - 4s 18ms/step - loss: 0.1514 - accuracy: 0.9438 - val_loss: 0.2427 - val_accuracy: 0.9173
Epoch 8/15
235/235 [==============================] - 4s 18ms/step - loss: 0.1311 - accuracy: 0.9510 - val_loss: 0.2462 - val_accuracy: 0.9185
Epoch 9/15
235/235 [==============================] - 4s 18ms/step - loss: 0.1125 - accuracy: 0.9576 - val_loss: 0.2628 - val_accuracy: 0.9184
Epoch 10/15
235/235 [==============================] - 4s 19ms/step - loss: 0.0995 - accuracy: 0.9630 - val_loss: 0.2543 - val_accuracy: 0.9253
Epoch 11/15
235/235 [==============================] - 4s 18ms/step - loss: 0.0833 - accuracy: 0.9678 - val_loss: 0.2838 - val_accuracy: 0.9170
Epoch 12/15
235/235 [==============================] - 4s 18ms/step - loss: 0.0679 - accuracy: 0.9747 - val_loss: 0.2931 - val_accuracy: 0.9209
Epoch 13/15
235/235 [==============================] - 4s 18ms/step - loss: 0.0592 - accuracy: 0.9787 - val_loss: 0.3195 - val_accuracy: 0.9223
Epoch 14/15
235/235 [==============================] - 4s 18ms/step - loss: 0.0466 - accuracy: 0.9831 - val_loss: 0.3672 - val_accuracy: 0.9220
Epoch 15/15
235/235 [==============================] - 4s 18ms/step - loss: 0.0427 - accuracy: 0.9842 - val_loss: 0.3951 - val_accuracy: 0.9169
</pre>
