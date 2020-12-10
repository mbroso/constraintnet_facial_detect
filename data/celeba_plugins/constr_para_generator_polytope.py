"""This file provides functionality for sampling constraint parameters for
2d-polytopes randomly located around a specific landmark. This sampling procedure
can be specified via the option constr_para_generator = 'opts2lm_polytope_rand'.
The generated constraint parameter for a 2d-polytope with n vertices consist of 2*n
parameters. 
"""

import torch
import math
import numpy as np
from pathlib import Path
import time


def opts2lm_polytope_rand(opts):
    """Creates LmPolytopeRand object by calling its constructor with option from
    opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        lm_polytope_rand (obj): Instantiated LmPolytopeRand object
    """

    return LmPolytopeRand(opts.lm_polytope_area, opts.lm_polytope_lm_class, 
            opts.lm_polytope_n, opts.lm_polytope_limits_delta_r, 
            opts.lm_polytope_limits_delta_angle)


class LmPolytopeRand:
    """This class implements a functor for sampling constraint parameters for
    2d-polytopes with n=3,4,5 vertices.

    The functor takes a sample of data as input and ouputs a torch tensor with
    generated vertice coordinates of the polytope [x0, y0, x1, y1, ..., xn,
    yn]. For construction, an equilateral 2d-polytope is constructed around a
    specified landmark and for randomization the deviations of the polar
    coordinates (radius and angle) of the vertices are sampled. 
    """

    def __init__(self, area, lm_class, n, limits_delta_r, limits_delta_angle):
        """Constructor for setting options.

        Args:
            area (int): Specifies the covered area of the equilateral
                2d-polytope when the sampled deviations of the polar coordinates
                are zero.
            lm_class (str): Defines the landmark the polytope is constructed
                around.
            n (int): number of vertices (e.g. 3 -> triangle)
            limits_delta_r (list): List with 2 elements. The first element
                specifies the minimal value when sampling radius deviations. The
                second element specifies the maximal value when sampling radius
                deviations.
            limits_delta_angle (list): List with 2 elements. The first element
                specifies the minimal value when sampling angle deviations. The
                second element specifies the maximal value when sampling angle
                deviations. The angles are measured in degree.
        
        """

        self.delta_r_min = limits_delta_r[0]
        self.delta_r_max = limits_delta_r[1]

        self.area = area
        self.lm_class = lm_class

        self.delta_angle_min = limits_delta_angle[0]
        self.delta_angle_max = limits_delta_angle[1]

        self.n = n
        # radius depending on the area for construction of equilateral 2d
        # polytope
        if n == 3:
            alpha = 60
            self.radius = np.sqrt(area)*np.sqrt(4./(3.*np.sqrt(3)))
        elif n == 4:
            self.radius = math.sqrt(area / 2)
        elif n == 5:
            self.radius = math.sqrt(area / 2.377)
        else:
            ValueError("2d-polytope with {n} vertices is not supported".format(n=n))
        self.delta_phi = 360 / self.n

    def __call__(self, sample):
        """Functor for sampling the constraint parameter of a 2d polytope.

        Args:
            sample (dict): Dictionary with position of the specified landmark 
                in sample[self.lm_class].

        Returns:
            constr_para (obj): Torch tensor with constraint parameter. The
                constraint parameter is given by the vertices of the polytope
                [x0, y0, x1, y1, x2, y2, x3, y3].
        """

        x_lm = sample[self.lm_class][0]
        y_lm = sample[self.lm_class][1]

        constr_para = torch.Tensor(int(2 * self.n))

        for i in range(int(self.n)):
            rand_r = torch.randint(int(self.delta_r_min), 
                    int(self.delta_r_max), (1,)) + self.radius

            #rand_r = np.random.uniform(self.delta_r_min, self.delta_r_max, (1,)) + self.radius 
            rand_angle = torch.randint(int(self.delta_angle_min), 
                    int(self.delta_angle_max), (1,)) 
            #rand_angle = np.random.uniform(self.delta_angle_min, self.delta_angle_max, (1,))
            if rand_r < 0:
                rand_r = 2.
            constr_para[2 * i] = x_lm + float(rand_r) * math.cos(math.radians(i *
                self.delta_phi - 90 + float(rand_angle)))
            constr_para[2 * i + 1] = y_lm + float(rand_r) * math.sin(math.radians(i * 
                self.delta_phi - 90 + float(rand_angle)))

        return constr_para

    @staticmethod
    def area_triangle(constr_para):
        """Computes the area of triangles encoded in constr_para.

        Args:
            constr_para (obj): Tensor (k, 6) with vertices coordinates of
                triangle.

        Returns:
            area (np.array): Array (k,) with areas of triangles.
        """
        constr_para = constr_para.data.numpy()
        l_1 = np.sqrt((constr_para[:, 2] - constr_para[:, 0])**2 +
                (constr_para[:, 3] - constr_para[:, 1])**2)
        l_2 = np.sqrt((constr_para[:, 4] - constr_para[:, 2])**2 +
                (constr_para[:, 5] - constr_para[:, 3])**2)
        l_3 = np.sqrt((constr_para[:, 0] - constr_para[:, 4])**2 +
                (constr_para[:, 1] - constr_para[:, 5])**2)
        l = (l_1 + l_2 + l_3) / 2
        area = np.sqrt( l * (l - l_1) * (l - l_2) * (l - l_3))
        return area


