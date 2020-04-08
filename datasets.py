"""Utilities for creating and manipulating Tensorflow 2 Datasets.

All datasets below return features as column tensors with shape
(num_attributes, 1) and labels as tensors with shape (1, 1)

Author: Nathan Sprague
Version: 3/2/2020

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import os.path
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def numpy_to_dataset(features, labels, shuffle=True):
    """ Convert numpy arrays to a tf.data.Dataset. 

    Args:
      features (ndarray): numpy array with shape (num_elements, num_attributes)
      labels (ndarray): integer numpy array with shape (num_elements,)
      shuffle (boolean): indicates whether the resulting dataset should
                         automatically shuffle after each epoch.

    Returns:
      tf.data.Dataset: Dataset that generates tuples containing
                       (feature, label) pairs, where feature and label
                       are tensors. features have shape (num_attributes, 1)
                       labels have shape (1, 1)
    """
    
    feature_dataset = tf.data.Dataset.from_tensor_slices(features)
    feature_dataset = feature_dataset.map(lambda x: tf.reshape(x, (-1, 1)))
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    label_dataset = label_dataset.map(lambda x: tf.reshape(x, (1, 1)))
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))

    if shuffle:
        dataset = dataset.shuffle(labels.shape[0],
                                  reshuffle_each_iteration=True)
    return dataset

def dataset_to_numpy(dataset):
    """Convert an appropriately structured tf.data.Dataset to a pair
    of numpy arrays.

    Args:
      dataset (tf.data.Dataset): Dataset that generates tuples containing
                       (feature, label) pairs, where feature and label
                       are tensors. features have shape (num_attributes, 1)
                       labels have shape (1, 1)

    Returns: features, labels: features is numpy array with shape
      (num_elements, num_attributes), labels is a numpy array with
      shape (num_elements,).n

    """
    dataset_np =  tuple(tfds.as_numpy(dataset))
    features = np.array([x[0].flatten() for x in dataset_np])
    labels = np.array([x[1] for x in dataset_np]).reshape((-1,))
    return features, labels

def two_clusters(num_points, noise=.3):
    """ Synthetic two-class dataset. """
    features = np.random.randint(2, size=(num_points, 1))
    features = np.append(features, features, axis=1)
    labels = np.array(np.logical_or(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + np.random.normal(0, noise, features.shape),
                        dtype=np.float32)

    # import matplotlib.pyplot as plt
    # plt.plot(features[labels==1, 0], features[labels==1, 1], 's')
    # plt.plot(features[labels==0, 0], features[labels==0, 1], 'o')
    # plt.show()

    dataset = numpy_to_dataset(features, labels)
    return dataset

def noisy_xor(num_points):
    """ Synthetic Dataset that is not linearly separable. """

    features = np.random.randint(2, size=(num_points, 2))
    labels = np.array(np.logical_xor(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + (np.random.random(features.shape) - .5),
                        dtype=np.float32)

    dataset = numpy_to_dataset(features, labels)

    # import matplotlib.pyplot as plt
    # plt.plot(features[labels==1, 0], features[labels==1, 1], 's')
    # plt.plot(features[labels==0, 0], features[labels==0, 1], 'o')
    # plt.show()

    return dataset

def make_mnist_binary(split='train'):
    """Binary version of the mnist dataset. Includes only classes 0 and
    1.

    """
    if not os.path.exists(split + 'labels.npy'):
        dataset = tfds.load(name="mnist", split=split)
        ds_numpy = tfds.as_numpy(dataset)

        images = np.array([ex['image'] for ex in ds_numpy])
        dataset = tfds.load(name="mnist", split=split)
        ds_numpy = tfds.as_numpy(dataset)
        labels = np.array([ex['label'] for ex in ds_numpy])

        zero_indices = labels == 0
        one_indices = labels == 1

        num_zeros = np.sum(zero_indices)
        num_ones = np.sum(one_indices)

        num_train = num_zeros + num_ones

        result_labels = np.empty((num_train,))
        result_images = np.empty([num_train] + list(images.shape[1::]))

        result_labels[0:num_zeros] = labels[zero_indices]
        result_labels[num_zeros::] = labels[one_indices]

        result_images[0:num_zeros, ...] = images[zero_indices]
        result_images[num_zeros::, ...] = images[one_indices]
        np.save(split + 'labels.npy', result_labels)
        np.save(split + 'images.npy', result_images)
    else:
        result_labels = np.load('labels.npy')
        result_images = np.load('images.npy')

    # Rescale and convert to floats...
    result_images = np.array(result_images / 255.0, dtype=np.float32)
    result_labels = np.array(result_labels, dtype=np.float32)

    dataset = numpy_to_dataset(result_images, result_labels)

    return dataset


if __name__ == "__main__":
    #make_mnist_binary()
    #make_noisy_xor(1000)
    two_clusters(1000)
