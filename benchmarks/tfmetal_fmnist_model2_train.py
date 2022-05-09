import tensorflow as tf
import tensorflow_datasets as tfds
import time

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
batch_size =256 

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
  tf.keras.layers.Conv2D(196, (2, 2), strides=(1,1), activation='relu', name='Input-conv2d'),
  tf.keras.layers.Conv2D(256, (2, 2), strides=(1,1), activation='relu', name='L2-conv2d'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L3-MaxP'),
  tf.keras.layers.Conv2D(256, (1, 1), activation='relu', name='L4-conv2d'),
  tf.keras.layers.Conv2D(512, (2, 2), activation='relu', name='L5-conv2d'),
  tf.keras.layers.Dropout(0.1589, name='L6-Drop'),
  tf.keras.layers.Flatten(name='L7-flat'),
  tf.keras.layers.Dense(128, activation='relu', name='L8-Dense'),
  tf.keras.layers.Dropout(0.5683, name='L9-Drop'),
  tf.keras.layers.Dense(10, activation='softmax', name='Output-Dense')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
 # changing the epochs count doesn't affect total memory used, but it does improve accuracy
start_time = time.time()
model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)
end_time = time.time()

print('Test loss: ', model.loss)
print('Elapsed Time: %0.4f seconds' %(end_time - start_time))
print('Elapsed Time: %0.4f' % ((end_time - start_time)/60))
print(model.summary())