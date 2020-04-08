"""Very simple binary classifiers using low-level tensorflow.  This
isn't intended to be a practical implementation, it is intended as an
illustration of TensorFlow 2 fundamentals.

Author: Nathan Sprague && Brady Imler
Version: 3/1/2020

"""

# Tensorflow tends to produce non-informative warnings.  This silences them.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.metrics
import abc
import datasets

class AbstractTFClassifier(abc.ABC):
    """ Abstract base class for simple TensorFlow-based classifiers. """

    @staticmethod
    def logistic(z):
        """ Calculate the logistic function elementwise.

        Args:
            z (tensor)
        Return:
           tensor with the same shape as z
        """
        return 1. / (1 + tf.exp(-z))

    @staticmethod
    def relu(z):
        """ Calculate the logistic function elementwise.

        Args:
            z (tensor)
        Return:
           tensor with the same shape as z
        """
        return tf.math.maximum(0,z)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        """ Calculate the cross entropy loss elementwise.

        Args:
            y_true (tensor): actual value(s)
            y_pred (tensor): predicted value(s)
        Return:
           tensor with the same shape as the inputs.
        """

        return -(y_true * tf.math.log(y_pred) +
                 (1-y_true) * tf.math.log(1-y_pred))


    @abc.abstractmethod
    def predict(self, x):
        """ Predict real-valued probability for the provided input(s).

        Args:
            x (tensor): either a tensor or a batch of tensors with
                        the appropriate dimensionality
        Returns:
           tensor with shape shape (1, 1) or (batch_size, 1, 1)
        """
        pass

    def train(self, dataset, learning_rate=.001, epochs=10):
        """Train the classifier using SGD for the specified number of epochs.

        Args:
            dataset (tf.data.Dataset): A dataset that returns
                                       (feature, label) tuples. The dataset
                                       should be configured to shuffle.
            learning_rate (float): the learning rate.
            epochs (int): The number of times to iterate through the dataset
                          during training.

        """
        for epoch in range(epochs):
            total_loss = 0
            for feature_batch, label_batch in dataset:
                with tf.GradientTape(persistent=True) as tape:
                    for index in range(0,len(self.weight_list)):
                        tape.watch(self.weight_list[index])
                        tape.watch(self.bias_list[index])

                    y_hat = self.predict(feature_batch)
                    loss = self.cross_entropy(label_batch, y_hat)

                for index in range(0,len(self.weight_list)):
                    # Gradient returns the average gradient across the batch.
                    dloss_dw = tape.gradient(loss, self.weight_list[index])
                    dloss_db = tape.gradient(loss, self.bias_list[index])

                    # Take one gradient step:
                    self.weight_list[index] = self.weight_list[index]- learning_rate *  dloss_dw
                    #for input bias
                    if (dloss_db != None):
                        self.bias_list[index] = self.bias_list[index] - learning_rate *  dloss_db

                # Add up the loss for this batch:
                total_loss += tf.reduce_sum(loss, axis=0)

            print("Epoch {} loss: {}".format(epoch, total_loss))


    def score(self, dataset):
        """ Return the accuracy of this model on the provided dataset and
        print a confusion matrix.

        Args:
            dataset (tf.data.Dataset): A dataset that returns
                                       (feature, label) tuples.

        Returns:
            Accuracy as a float.
        """

        all_y_yat = []
        all_labels = []
        for feature_batch, label_batch in dataset:
            all_y_yat.append(self.predict(feature_batch).numpy())
            all_labels.append(label_batch.numpy())

        all_y_hat = np.array(np.round(all_y_yat), dtype=int).flatten()
        all_labels = np.array(all_labels, dtype=int).flatten()

        accuracy = sklearn.metrics.accuracy_score(all_labels, all_y_hat)

        print("Accuracy: {:.5f}".format(accuracy))
        print("Confusion Matrix")
        print(sklearn.metrics.confusion_matrix(all_labels, all_y_hat))
        return accuracy

    def plot_2d_predictions(self, dataset):
        """Plot the decision surface and class labels for a 2D dataset."""

        import matplotlib.pyplot as plt
        features, labels = datasets.dataset_to_numpy(dataset)

        minx = np.min(features[:, 0])
        maxx = np.max(features[:, 0])
        miny = np.min(features[:, 1])
        maxy = np.max(features[:, 1])

        x = np.linspace(minx, maxx, 100)
        y = np.linspace(miny, maxy, 100)
        xx, yy = np.meshgrid(x, y)
        fake_features = np.array(np.dstack((xx, yy)).reshape(-1, 2, 1),
                                 dtype=np.float32)
        z = self.predict(fake_features).numpy().reshape(len(x), len(y))
        plt.imshow(np.flipud(np.reshape(z,(len(x),len(y)))),
                   vmin=-.2, vmax=1.2,extent=(minx, maxx, miny, maxy),
                   cmap=plt.cm.gray)
        CS = plt.contour(x, y, z)
        plt.clabel(CS, inline=1)

        markers = ['o', 's']
        for label in np.unique(labels):
            plt.scatter(features[labels==label, 0],
                        features[labels==label, 1],
                        marker=markers[int(label)])

        plt.show()



class LogisticRegression(AbstractTFClassifier):
    """ Simple Logistic Regression Classifier """

    def __init__(self, input_dim):
        """ Build the classifier and initialize weights randomly.

        Args:
            input_dim (int): dimensionality of the input
        """
        super().__init__()
        b = tf.Variable(tf.random.uniform(minval=-.001,
                                               maxval=.001,
                                               shape=(1, 1)))
        w = tf.Variable(tf.random.uniform(minval=-.001,
                                               maxval=.001,
                                               shape=(input_dim, 1)))
        self.weight_list = [w]
        self.bias_list = [b]


    def predict(self, x):
        return self.logistic(tf.matmul(tf.transpose(self.weight_list[0]), x) + self.bias_list[0])




class MLP(AbstractTFClassifier):
    """ Simple three-layer neural network classifier."""

    def __init__(self, input_dim, num_hidden):
        super().__init__()

        #weights
        in_weights = tf.Variable(tf.random.normal(stddev=(1/tf.sqrt(float(3))),
                                               shape=(input_dim, num_hidden)),
                                               trainable=True)
        hidden_weights = tf.Variable(tf.random.normal(stddev=(1/tf.sqrt(float(num_hidden))),
                                               shape=(num_hidden, num_hidden)),
                                               trainable=True)
        out_weights = tf.Variable(tf.random.normal(stddev=(1/tf.sqrt(float(num_hidden))),
                                               shape=(num_hidden, 1)),
                                               trainable=True)
        #bias
        #input bias is not actually used
        in_bias = tf.Variable(tf.random.uniform(shape=(1,1)),
                                            trainable=True)
        hidden_bias = tf.Variable(tf.random.uniform(shape=(1,1)),
                                                        trainable=True)
        out_bias = tf.Variable(tf.random.uniform(shape=(1, 1)),
                                                        trainable=True)
        #put into lists for training function
        self.weight_list=[in_weights, hidden_weights, out_weights]
        self.bias_list=[in_bias, hidden_bias, out_bias]


    def predict(self, x):
        #no bias on input layer
        layer1 = self.relu(tf.matmul(tf.transpose(self.weight_list[0]),x))
        layer2 = self.logistic(tf.matmul(tf.transpose(self.weight_list[1]), layer1) + self.bias_list[1])
        layer3 = self.logistic(tf.matmul(tf.transpose(self.weight_list[2]), layer2) + self.bias_list[2])
        return layer3


if __name__ == "__main__":
    pass
