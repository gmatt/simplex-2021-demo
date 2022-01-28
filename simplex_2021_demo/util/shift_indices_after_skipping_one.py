from typing import List

import numpy as np


def shift_indices_after_skipping_one(indices: List[int], skipped_index: int):
    """
    The main solution consists of skipping a single example, getting a list of indices, then figuring out which indices
    correspond from the skipped list to the original one.
    This helper function does exactly that.

    >>> l = [1, 0, 0, 1]
    >>> skipped_index = 1
    >>> del l[skipped_index]
    >>> l
    [1, 0, 1]
    >>> indices = [0, 2]
    >>> shift_indices_after_skipping_one(indices, skipped_index)
    [0, 3]


    Args:
        indices:
        skipped_index:

    Returns:

    """
    x = np.array(indices)
    x = x + (x >= skipped_index)
    return x.tolist()
