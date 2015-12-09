""" Function for reading text file as utf-8 and return an array storing data in the correct format """
import os
import sys
import numpy as np
import copy
import codecs
import random as rd

def check_Python_version():
    """ This script runs upon module import to check
    for python version """
    major = sys.version_info[0]
    minor = sys.version_info[1]
    micro = sys.version_info[2]
    if not major == 2 :
        print('Please use Python 2.7 for now. Your Python is too fancy for me!')
        raise StandardError('Python version %d.%d.%d is not compatible!' % (major, minor, micro))
    else:
        if not minor == 7 :
            print('Your Python might work! But if it does not, use 2.7!')
            raise Warning('Use python 2.7 for the good stuffs!')

def is_there_file(filename, work_directory):
    """ Check if there is the required file """
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        raise IOError('File %s is not found' % filepath)
    return filepath

def extract_name_win(filename):
    """ Extract and return horse name and number of win as
    an array of tuples which have name as string of utf-8
    katakana and number of win as integer """
    check_Python_version()
    bamei = codecs.open(is_there_file(filename, os.getcwd()), 'r', 'utf-8')
    name_win = []
    for line in bamei:
        name, win = line.split(',', 2)
        name = name.replace('"','') # Get rid of double quotes
        win = int(win.replace('"','')) # Get rid of dobule quotes and \n
        name_win.append([name, win])
    """ This line is used for getting rid of the unicode bit order signature
    in the beginning of the file due to the codecs utf-8 read.
    TODO: Find a better solution. """
    name_win[0][0] = name_win[0][0][1:]
    return name_win

def string_to_decimal(data):
    """ Convert data from utf-8 katakana string to its
    decimal code point. Data = [ [name, win] ... ] """
    name = []
    for i in range(0, len(data)):
        name.append([ord(x) for x in data[i][0]])
    return name

def decimal_to_onehot(data):
    """ Convert decimal name data to onehot, return
    the data as array of one hot Numpy matrix """
    temp = copy.copy(data)
    one_hot = []
    # Get max and min char code
    max_name_len = 0;
    max_char_code = 0;
    min_char_code = 99999; # Magic number! :3
    for i in data:
        if max_name_len < len(i):
           max_name_len = len(i)
        for j in i:
            if (j > max_char_code):
                max_char_code = j
            if (j < min_char_code):
                min_char_code = j
    # Compute offset for each character
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            temp[i][j] = data[i][j] - min_char_code
    # Create one hot matrix array
    for i in temp:
        tempm = np.zeros(shape=(max_name_len,max_char_code-min_char_code+1))
        for j in range(0, len(i)):
           tempm[j][i[j]] = 1
        one_hot.append(tempm)
    return np.array(one_hot)

def trim_data(name, label, remove_chance = 0.9):
    """ Randomly remove 'remove_chance' useless data with label 0 """
    assert len(name) == len(label), 'Length of name and label is not match.'
    n_name = []
    n_label = []
    for i in range(0, len(name)):
        if label[i] == 0 :
            if not rd.random() > remove_chance :
                pass
            else :
                n_name.append(name[i])
                n_label.append(label[i])
        else :
            n_name.append(name[i])
            n_label.append(label[i])
    return np.array(n_name), np.array(n_label)

class HorseName(object):
    """ Class to contain data about horse name and number of wins """
    def __init__(self, names, labels, rand_data = False):
        if rand_data :
            self._num_examples = 10000
        else:
            assert names.shape[0] == labels.shape[0], ('Name.shape: %s, labels.shape: %s' % (names.shape, labels.shape))
            self._num_examples = names.shape[0]
            # Flatten names
            names = names.reshape(names.shape[0], names.shape[1]*names.shape[2])
            self._names = names
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

    @property
    def names(self):
        return self._names

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, rand_data = False):
        """ Function to get 'batch_size' number of training data for
        each epoch training session. """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples :
            # Finish epoch
            self._epoch_completed += 1
            # Shuffle data to get next batch
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._names = self._names[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._names[start:end], self._labels[start:end]

def read_data_set(train_dir, rand_data = False):
    """ Provide a data warper for the horse data set """
    class HorseDataSet(object):
        pass
    data_sets = HorseDataSet()
    if rand_data:
        data_sets.train = HorseName([],[],True)
        data_sets.validation = HorseName([],[],True)
        data_sets.test = HorseName([],[],True)
        return data_sets
    DATA_FILE = 'bamei_win_utf8.csv'
    TRIM_RATE = 0.8

    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 3000


    """ Read the data file and separate the test / train / validation """
    local_file = is_there_file(DATA_FILE,train_dir)
    name_win = extract_name_win(local_file)
    win = [i[1] for i in name_win]
    dec_name = string_to_decimal(name_win) # Convert utf-8 code point to decimal
    name = decimal_to_onehot(dec_name) # Convert decimal array to get 13 x 92 matrix one hot name
    name, win = trim_data(name, win, TRIM_RATE) # Remove useless horses

    # Attach label to horses 
    labels = []
	# Making discrete labels for each horses
    for i in win:
        if (i > 1):
            labels.append([0, 1])
        else:
            labels.append([1, 0])

    labels = np.array(labels)
    perm = np.arange(len(name))
    np.random.shuffle(perm)
    train_len = int(len(name) * TRAIN_SIZE)
    train_name = name[perm[:train_len]]
    train_label = labels[perm[:train_len]]
    valid_name = train_name[:VALIDATION_SIZE]
    valid_label = train_label[:VALIDATION_SIZE]
    test_name = name[perm[train_len:]]
    test_label = labels[perm[train_len:]]
    data_sets.train = HorseName(train_name, train_label)
    data_sets.validation = HorseName(valid_name, valid_label)
    data_sets.test = HorseName(test_name, test_label)

    return data_sets
