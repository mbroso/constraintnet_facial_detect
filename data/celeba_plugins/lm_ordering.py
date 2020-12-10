"""This file contain a y_generator which arrange landmark coordinates in a 
defined way.
"""

import torch
import numpy as np

def opts2lm_ordering(opts):
    """Creates LmOrdering functor by calling its constructor with parameters 
    from opts.

    Args:
        opts (obj): Namespace object returned by parser with settings.

    Returns:
        LmOrdering (obj): Instantiate LmOrdering functor.
    """
    return LmOrdering(opts.lm_ordering_lm_order)

class LmOrdering:
    """Functor to create the target/ground truth y by ordering landmark 
    coordinates.

    This functor gets a preprocessed sample and creates a y tensor with 
    landmark coordinates in defined order.
    """
    def __init__(self, lm_order):
        """Initialization by defining order of landmark coordinates in y.

        Args:
            lm_order (list): List with landmark coordinates. E.g. 
                [nose_x, nose_y]
        """
        self.lm_order = lm_order

    def __call__(self, sample):
        """Creates the target y for a preprocessed sample by arranging landmark
        coordinates.

        The landmark coordinates are arranged according to self.lm_order.

        Args:
            sample (dict): Preprocessed sample with keys: 'nose', ...

        Returns:
            y (obj): Torch tensor with shape (len(lm_order),) with landmarks
                in defined order.
        """
        #create tensor of shape (len(lm_order)) based on tensor of one landmark
        n = len(self.lm_order)
        one_lm, one_coord = self.lm_coord_split(self.lm_order[0])
        y = sample[one_lm].new(n)
        for i, lm_coord in enumerate(self.lm_order):
            lm, coord = self.lm_coord_split(lm_coord)
            if not lm in sample.keys():
                raise ValueError('Landmark {lm} is not in sample.'.format(
                    lm=lm)
                    )
            y[i] = sample[lm][coord]
        return y

    def lm_coord_split(self, lm_coord):
        """Creates the landmark keyword without coordinate prefix and the 
        coordinate index (x:0, y:1).

        E.g. lm_coord_split('nose_x') = 'nose', 0 

        Args:
            lm_coord (str): Landmark with coordinate postfix. E.g. 'nose_x'

        Returns:
            lm, coord (tuple): Tuple with landmark keyword (str, e.g 'nose')
                and landmark coordinate (int, 0 for x and 1 for y).
        """
        coord = 2
        if lm_coord[-2:]=='_x':
            coord = 0
        if lm_coord[-2:]=='_y':
            coord = 1
        if coord == 2:
            raise ValueError('Landmark coordinates must end with _x or _y.')
        lm = lm_coord[:-2]
        return lm, coord



