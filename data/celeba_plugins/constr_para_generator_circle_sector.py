"""This file provides functionality for sampling constraint parameters for
sectors of a circle randomly located around a specific landmark. This sampling 
procedure can be specified via the option constr_para_generator =
'opts2lm_circle_sector_rand'. The generated constraint parameter for the sector
of a circle is given by [x_c, y_c, R, Psi] with (x_c,y_c) the center position, R
the radius and Psi the angular range. Note that we consider symmetric sectors of
a circle with respect to the vertical line. 
"""


import torch
import math
import numpy as np
from pathlib import Path
import time


def opts2lm_circle_sector_rand(opts):
    """Creates LmCircleSectorRand object by calling its constructor with
    options from opts.

    Args:
        opts(obj): Namespace object with options.

    Returns:
        lm_circle_sector_rand(obj): Instantiated LmCircleSectorRand object

    """
    return LmCircleSectorRand(opts.lm_circle_sector_r_min,
            opts.lm_circle_sector_r_max, opts.lm_circle_sector_psi_min,
            opts.lm_circle_sector_psi_max, opts.lm_circle_sector_lm_class)


class LmCircleSectorRand:
    """This class implements a functor for sampling the constraint parameters
    for symmetric sectors of a circle.

    The functor takes a sample of data as input and ouputs a torch tensor with
    the constraint parameter  [x_c, y_c, R, Psi].
    """

    def __init__(self, r_min, r_max , psi_min, psi_max, lm_class):
        """Initialization for setting params of functor.

        Args:
            r_min (float): Minimal radius of sector of circle.
            r_max (float): Maximal radius of sector of circle.
            Psi_min (float). Minimal angle of sector of circle.
            Psi_max (float). Maximal angle of sector of circle.
        """
        self.r_min = r_min
        self.r_max = r_max
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.lm_class = lm_class


    def __call__(self, sample):
        """Generates constraint parameters for sectors of circle randomly
        located around a specific landmark.

        Args:
            sample (dict): Dictionary with position of the specified landmark 
                in sample[self.lm_class].

        Returns:
            constr_para (obj): Torch tensor with constraint parameter: [x_c,
            y_c, R, Psi]
        """

        x_lm = float(sample[self.lm_class][0])
        y_lm = float(sample[self.lm_class][1])
        constr_para = torch.Tensor(4)
        
        r = float(torch.randint(int(self.r_min*1000), int(self.r_max*1000), (1,)))/1000.
        #factor 100 for discretization
        psi = float(torch.randint(int(self.psi_min*1000), int(self.psi_max*1000),
                (1,)))/1000.
        y_c = float(torch.randint(int((y_lm - r)*1000), int((y_lm)*1000), (1,)))/1000.
        d_x_c = np.tan(psi/2)*(y_lm - float(y_c))
        print(psi, int(self.psi_min*1000), int(self.psi_max*1000))
        x_c_min = x_lm - d_x_c
        x_c_max = x_lm + d_x_c
        if int(x_c_min*1000) == int(x_c_max*1000):
            x_c = x_lm
        else:
            x_c = float(torch.randint(int(x_c_min*1000), int(x_c_max*1000), (1,)))/1000.

        constr_para[0] = x_c
        constr_para[1] = y_c
        constr_para[2] = r
        constr_para[3] = psi
        
        return constr_para

    @staticmethod
    def area(constr_para):
        """Computes the area covered by the sector of the circle.

        Args:
            constr_para (obj): Torch tensor with shape (N,4). Tensor represents
                the constraint parameter [x_c, y_c, R, Psi].

        Returns:
            area (np.array): Shape (N), area covered by the sector of a circle
                constraint.
        """

        r = constr_para[:, 2].data.numpy()
        psi = constr_para[:, 3].data.numpy()
        area = r**2 * math.pi * psi / (2*math.pi)
        return area
