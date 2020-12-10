"""This file provides functionality for sampling constraint parameters for
bounding boxes around the face. This sampling procedure can be specified via the
option constr_para_generator = 'opts2face_bb'. The generated constraint
parameter for the bounding box is given by the boundaries [l_x, u_x, l_y, u_y]. 
"""

import torch
import numpy as np
from pathlib import Path
import time


def opts2face_bb_rand(opts):
    """Creates FaceBbRand functor by calling its constructor with options
    from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        face_bb_rand (obj): Instantiated FaceBbRand functor.
    """
    return FaceBbRand(opts.face_bb_rand_padding_min, 
            opts.face_bb_rand_padding_max)


class FaceBbRand:
    """Functor to create constraint parameters for randomly generated bounding
    boxes around the face.  

    The bounding box is created by first finding the smallest rectangle which
    covers the eyes and the nose. In a second step some random padding is 
    added.
    """
    def __init__(self, padding_min, padding_max):
        """Initialization.

        Args:
            padding_min (int, list): Minimal value which is added to the 
                boundaries in x and y direction. If int, minimal padding value 
                for x and y is the same. If list, minimal padding value in x 
                and y direction are settable separately. Padding is added by 
                random value between padding_min and padding_max.
            padding_max (int, list): Analog to padding_min for the maximal 
                padding values.
        """
        self.padding_min = padding_min
        if isinstance(self.padding_min, int):
            self.padding_min = [self.padding_min, self.padding_min]
        if len(self.padding_min)==1:
            value = self.padding_min[0]
            self.padding_min = [value, value]
        
        self.padding_max = padding_max
        if isinstance(self.padding_max, int):
            self.padding_max = [self.padding_max, self.padding_max]
        if len(self.padding_max)==1:
            value = self.padding_max[0]
            self.padding_max = [value, value]
        

    def __call__(self, sample):
        """Creates a torch tensor for the constraint parameter.

        Args:
            sample (obj): Dictionary with keys 'img' and specified landmarks. 
                Must contain landmarks nose, lefteye and righteye.
        
        Returns:
            constr_para (obj): Torch tensor which contains the bounding box 
                coordinates as parameters: (l_x, u_x, l_y, u_y).
        """
        #img: C x H x W
        H = sample['img'].shape[1]
        W = sample['img'].shape[2]
        
        nose = sample['nose']
        lefteye = sample['lefteye']
        righteye = sample['righteye']
        x_list = [nose[0], lefteye[0], righteye[0]]
        y_list = [nose[1], lefteye[1], righteye[1]]

        constr_para = torch.Tensor(4)
        
        #find first the minimal rectangle covering nose and eye lms
        l_x = min(x_list)
        u_x = max(x_list)
        l_y = min(y_list)
        u_y = max(y_list)

        #add some random padding to rectangle boundaries
        left = torch.randint(int(self.padding_min[0]), int(self.padding_max[0]), (1,) )
        l_x = l_x - left  
        right = torch.randint(int(self.padding_min[0]), int(self.padding_max[0]), (1,) )
        u_x = u_x + right
        upper = torch.randint(int(self.padding_min[1]), int(self.padding_max[1]), (1,) )
        u_y = u_y + upper
        lower = torch.randint(int(self.padding_min[1]), int(self.padding_max[1]), (1,) )
        l_y = l_y - lower

        #clip if exceeds img boundaries
        l_x = max([0, l_x])
        u_x = min([W-1., u_x])
        l_y = max([0, l_y])
        u_y = min([H-1., u_y])
        
        constr_para[0] = l_x
        constr_para[1] = u_x
        constr_para[2] = l_y
        constr_para[3] = u_y
        
        return constr_para




