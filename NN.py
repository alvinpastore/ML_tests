__author__ = 'alvin'
import matplotlib.pyplot as plt
import gzip
import cPickle
import numpy as np

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train, valid_set, test_set = cPickle.load(f)

train_set = train[0]
train_labels = train[1]

for vec in train_set:
    print vec.shape
    raw_input()
    vec = np.reshape(vec, (28, 28))
    print vec
    print 'shape' + str(np.shape(vec))
    plt.imshow(vec)
    plt.show()
    raw_input()