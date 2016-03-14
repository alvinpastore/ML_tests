import matplotlib.pyplot as plt
import gzip
import cPickle
import numpy as np

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train, valid_set, test_set = cPickle.load(f)

train_set = train[0]
train_labels = train[1]

for vec in train_set:
    plt.imshow(vec.reshape((28, 28)), interpolation='nearest')
    plt.show()
