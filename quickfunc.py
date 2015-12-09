""" Quick template function for drafting HorseNameNN """

_epoch_completed = 0

def next_batch(data, batch_size):
    """ next_batch takes the data inform of [sample, label]
    where sample is array of feature vectors and label is
    one-hot vector for classification. batch_size is the number
    of samples to put in the neural network for the current
    training section. """
    start =

