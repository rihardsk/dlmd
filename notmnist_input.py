from six.moves import cPickle as pickle
import numpy as np

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


print(np.bincount(test_labels), np.bincount(test_labels).std())
print(np.bincount(valid_labels), np.bincount(valid_labels).std())

image_size = 28
num_labels = 10
num_channels = 1

# def whiten(images):
#     pass

def reformat(dataset, labels):
    reshaped_dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # reshaped_dataset = dataset.astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    one_hot_labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return reshaped_dataset, one_hot_labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)