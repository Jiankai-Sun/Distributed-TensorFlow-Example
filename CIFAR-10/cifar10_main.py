"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import horovod.tensorflow as hvd

import tensorflow as tf

import resnet_model

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(is_training, data_dir))

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def cifar10_model_fn(features, labels, params):
  """Model function for CIFAR-10."""
  tf.summary.image('images', features, max_outputs=6)

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format'])

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, is_training=True)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  # Scale the learning rate linearly with the batch size. When the batch size
  # is 128, the learning rate should be 0.1.
  initial_learning_rate = 0.1 * params['batch_size'] / 128
  batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
  global_step = tf.train.get_or_create_global_step()

  # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
  values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
  learning_rate = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)

  # Create a tensor named learning_rate for logging purposes
  tf.identity(learning_rate, name='learning_rate')
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=_MOMENTUM)

  # Add Horovod Distributed Optimizer.
  optimizer = hvd.DistributedOptimizer(optimizer)

  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return train_op,loss,global_step


def main(unused_argv):
  # Initialize Horovod.
  hvd.init()

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  
  custom_params={
      'resnet_size': FLAGS.resnet_size,
      'data_format': FLAGS.data_format,
      'batch_size': FLAGS.batch_size,
    }

  features, labels = input_fn(True, FLAGS.data_dir, FLAGS.batch_size, None) 
  train_op, loss, global_step = cifar10_model_fn(features, labels, custom_params)

  # BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0
  # to all other processes. This is necessary to ensure consistent initialization
  # of all workers when training is started with random weights or restored
  # from a checkpoint.
  hooks = [hvd.BroadcastGlobalVariablesHook(0),
            tf.train.StopAtStepHook(last_step=10000),
            tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                    every_n_iter=10),
            ]

  # Pin GPU to be used to process local rank (one GPU per process)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())

  # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
  checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

  # The MonitoredTrainingSession takes care of session initialization,
  # restoring from a checkpoint, saving to a checkpoint, and closing when done
  # or an error occurs.
  with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                        hooks=hooks,
                                        config=config) as mon_sess:
    while not mon_sess.should_stop():
        # Run a training step synchronously.
        mon_sess.run(train_op) 

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
