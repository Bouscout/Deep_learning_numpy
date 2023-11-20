# some utility function
import random
from math import floor
import numpy as np


# an array shuffler :
def shuffler(arr1, arr2):
    current_index = len(arr1)
    random_index = len(arr1)
    # while there are still elements to shuffle
    while current_index != 0 :
        random_index = floor((random.random() * current_index)) 
        current_index -= 1

        # swapping them
        arr1[current_index], arr1[random_index] = arr1[random_index], arr1[current_index]

        # swapping the y arrays with the same index
        arr2[current_index], arr2[random_index] = arr2[random_index], arr2[current_index]

    return (arr1, arr2)

# divide input and label into batches itterator
def divide_batch(x_data: np.ndarray, y_data: np.ndarray, batch_size: int):
    size = x_data.shape[0]
    batch = batch_size

    while batch <= size:
        index = batch - batch_size 
        yield (x_data[index:batch], y_data[index:batch])
        batch += batch_size

    last = batch - batch_size
    if last < size:
        yield (x_data[last:size], y_data[last:size])