#!/usr/bin/python
"""
Examples of running tf_classifers.
"""

import tf_classifiers
import datasets

def logistic_regression_clusters():
    dataset_train = datasets.two_clusters(500)
    dataset_train = dataset_train.batch(100)
    dataset_test = datasets.two_clusters(500)
    classifier = tf_classifiers.LogisticRegression(2)
    classifier.train(dataset_train, epochs=20, learning_rate=.01)
    classifier.score(dataset_test)
    classifier.plot_2d_predictions(dataset_test)

def logistic_regression_xor():
    dataset_train = datasets.noisy_xor(500)
    dataset_train = dataset_train.batch(100)
    dataset_test = datasets.noisy_xor(500)
    classifier = tf_classifiers.LogisticRegression(2)
    classifier.train(dataset_train, epochs=20, learning_rate=.01)
    classifier.score(dataset_test)
    classifier.plot_2d_predictions(dataset_test)

def MLP_clusters():
    dataset_train = datasets.two_clusters(500)
    dataset_train = dataset_train.batch(100)
    dataset_test = datasets.two_clusters(500)
    classifier = tf_classifiers.MLP(2,5)
    classifier.train(dataset_train, epochs=20, learning_rate=.01)
    classifier.score(dataset_test)
    classifier.plot_2d_predictions(dataset_test)

def MLP_xor():
    dataset_train = datasets.noisy_xor(500)
    dataset_train = dataset_train.batch(100)
    dataset_test = datasets.noisy_xor(500)
    classifier = tf_classifiers.MLP(2,30)
    classifier.train(dataset_train, epochs=100, learning_rate=.01)
    classifier.score(dataset_test)
    #classifier.plot_2d_predictions(dataset_test)


if __name__ == "__main__":
    #logistic_regression_xor()
    #logistic_regression_clusters()
    MLP_xor()
    #MLP_clusters()
