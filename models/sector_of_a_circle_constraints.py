"""This file summarizes functionality for the construction of ConstraintNet with
output-constraints in form of sectors of a circle.

The functionality for modelling the output-constraints consists namely of: 

    - Functors to create a tensor representation g(s) of the constraint
        parameter s. The functor can be selected via the option
        opts.opts2constr_para_repr. E.g. opts.opts2constr_para_repr =
        'opts2const_feat_planes' would call the function opts2const_feat_planes
        which instantiate the functor ConstFeatPlanes. ConstFeatPlanes is then
        used for g(s). 
    - Functors to tranform the constraint parameter s into a vertex
        representation. The vertex representation consists of vertices which
        describe the convex polytope(s). The functor can be selected via the option
        opts.opts2constr_para_trf. E.g. opts.opts2constr_para_trf =
        'opts2identity' would call
        the function opts2identity which instantiate the functor Identity.
        Identity lets the constraint parameter unchanged and is used to match
        the interface.
    - PyTorch modules for the constraint-guard layer, i.e. the mapping from the
        intermediate representation z to the constrained output region. The
        module can be selected via the option opts.opts2constr_guard_layer. For
        the considered sectors of a circle constraints in this file, the PyTorch
        module "SectorsOfACircle" can be selected via
        opts.opts2constr_guard_layer = 'opts2sectors_of_a_circle'. 
        
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def opts2const_feat_planes(opts):
    """Creates ConstFeatPlanes functor by calling its constructor with 
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        const_feat_planes (obj): Instantiated ConstFeatPlanes functor.
    """
    return ConstFeatPlanes(
            opts.const_feat_planes_h,
            opts.const_feat_planes_w,
            opts.const_feat_planes_n_channels,
            opts.const_feat_planes_repeat_channels,
            opts.const_feat_planes_norm_factor
            )


class ConstFeatPlanes:
    """This functor generates a tensor representation g(s) of the constraint
    parameter s. 

    Each component of the constraint parameter s corresponds to
    <repeat_channels> channels of the generated tensor. The assigned channels
    have entries with a constant value and are given by a constraint parameter
    rescaled with a factor. The height and width of the tensor can be specified.
    """
        
    def __init__(self, h, w, n_channels, repeat_channels=1, norm_factor=1.):
        """Initialization for setting parameters.

        Args:
            h (int): Height of channels.
            w (int): Width of channels.
            n_channels (int): Number of channels of generated tensor. 
                Specified number must match
                n_channels = <length of  constraint parameter) * repeat_channels
            repeat_channels (int): The channel for a constraint parameter can be
                replicated <repeat_channels> times.
            norm_factor (float or list): Factor to normalize the values of the 
                tensor. If float, all constraint parameter components are
                rescaled with this factor. If list, the length of the list must
                match the number of constraint parameter components and the list
                elements must be of type float. Each constraint parameter
                component is then rescaled with the corresponding factor in the
                list.
        """
        self.h = h
        self.w = w
        self.n_channels = n_channels
        
        self.repeat_channels = repeat_channels
        
        #extract number of constraint parameter components
        n_constr_para = int(n_channels / repeat_channels)
        if not n_channels == n_constr_para * repeat_channels:
            raise ValueError('Number of channels in constraint parameter \
                    tensor representation must be a \
                    multiple of repeat_channels. But n_channels={n_channels} \
                    and repeat_channels={repeat_channels}'.format(
                        n_channels = n_channels,
                        repeat_channels = repeat_channels)
                    )
        
        #convert norm_factor scalar in list format
        self.norm_factor = norm_factor

        if isinstance(self.norm_factor, float):
            norm_factor_value = self.norm_factor
            self.norm_factor = []
            for i in range(n_constr_para):
                self.norm_factor.append(norm_factor_value)
        if len(self.norm_factor)==1:
            norm_factor_value = self.norm_factor[0]
            self.norm_factor = []
            for i in range(n_constr_para):
                self.norm_factor.append(norm_factor_value)


        if not len(self.norm_factor) * repeat_channels == n_channels:
            raise ValueError('Number of norm factors for constr_para must \
                    match n_channels / repeat_channels. But \
                    len(norm_factor)={len_norm} is not equal to \
                    n_channels / repeat_channels = {n_channels} / \
                    {repeat_channels}'.format(
                        len_norm = len(self.norm_factor),
                        n_channels = n_channels,
                        repeat_channels = repeat_channels)
                    )


    def __call__(self, constr_para):
        """Functor to create tensor representation g(s).

        Args:
            constr_para (obj): Pytorch tensor of shape (N, n_constr_para)
                which specifies the output-constraint. 

        Returns:
            constr_para_repr (obj): Pytorch tensor for tensor representation
                g(s) of the constraint parameter with shape 
                (N, c_constr_para_repr, H, W).
        """

        if not self.n_channels == constr_para.shape[1] * self.repeat_channels:
            raise ValueError('Number of channels of the tensor representation \
                        of the constraint parameter must match with  \
                        the number of constraint parameter components times \
                        repeat_channel. But n_channels={n_channels} is \
                        not equal to n_constr_para * repeat_channels = \
                        {n_constr_para} * {repeat_channels}'.format(
                            n_channels = self.n_channels,
                            n_constr_para = constr_para.shape[1],
                            repeat_channels = self.repeat_channels)
                        )


        #create region feature tensor of correct shape
        constr_para_repr = constr_para.new(
                constr_para.shape[0], 
                self.n_channels, 
                self.h,
                self.w
                )

        #fill the constr_features tensor with the normed constr_para
        for i, sample in enumerate(constr_para):
            for j, para in enumerate(sample):
                j_in = j * self.repeat_channels
                for l in range(self.repeat_channels):
                    constr_para_repr[i, j_in + l,:,:] = para * self.norm_factor[j]
        
        return constr_para_repr



def opts2identity(opts):
    """Creates Identity functor by calling its constructor with options
    from opts. The Identity functor passes the constraint parameter unchanged
    and is used to match the interface.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        identity (obj): Instantiated Identity functor.

    """
    return Identity()

class Identity:
    """This functor passes the constraint parameter unchanged and is used to
    match the interface.
    """

    def __init__(self):
        pass

    def __call__(self, constr_para):
        """This functor gets the constraint parameter and returns the constraint
        parameter unchanged.

        Args:
            constr_para (tensor): Torch tensor containing the constraint
                parameter.

        Returns:
            constr_para (tensor): Torch tensor containing the constraint
                parameter.        
        """

        return constr_para


def opts2sectors_of_a_circle(opts):
    """Creates SectorsOfACircle nn.Modules by calling its constructor with
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        SectorsOfACircle (obj): Instantiated SectorsOfACircle nn.Module.
    """

    return SectorsOfACircle()

class SectorsOfACircle(nn.Module):
    """Constraint-guard layer for constraints in form of sectors of a circle.
    """
    def __init__(self):
        super(SectorsOfACircle, self).__init__()

    def forward(self, z, constr_para):
        """
        Args:
            z (tensor): Torch tensor with intermediate variable z with shape
                (N,2).
            constr_para (tensor): Pytorch tensor for constraint parameter of a
                sector of a circle with shape (N, 4). The four entries per
                sample are ordered according to (x,y,R,PSI).
        Returns:
           out (obj): Torch tensor with shape (N, 2). Each output is
                within circle segment specified by circle_segment.

        """
        p = torch.sigmoid(z)
        x = constr_para[:, 0]
        y = constr_para[:, 1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out = torch.Tensor(constr_para.shape[0], 2).to(device) 
        radius = constr_para[:, 2] * p[:, 1] 
        phi = constr_para[:, 3] * (p[:, 0] - 0.5)
        out[:, 0] = radius * torch.sin(phi) + x
        out[:, 1] = radius * torch.cos(phi) + y
        return out


