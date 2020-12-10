import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


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


def opts2bb_projection(opts):
    """Creates BbProjection nn.Modules by calling its constructor with
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        BbProjection (obj): Instantiated BbProjection nn.Module.
    """

    return BbProjection()

class BbProjection(nn.Module):
    """This is a functor which performs the projection for bounding box
    constraints. The package cvxpy is used.
    """
    def __init__(self):
        super(BbProjection, self).__init__()

    def __call__(self, y_pred, constr_para):
        """Projects y_pred on region defined by constr_para.

        Args:
            y_pred (obj): Pytorch tensor with shape (batch_size, 6). And the
                second dimension is for the landmark positions: nose_x,
                lefteye_x, righteye_x, lefteye_y, righteye_y, nose_y.
            constr_para (obj): Pytorch tensor with shape (batch_size, 4). And
                the second dimension is for the bounding box boundaries l_x,
                u_x, l_y, u_y.

        Returns:
            y_proj (obj): Pytorch tensor with projected version of y_pred.
        """
        y_proj = y_pred.new(y_pred.shape)
        batch_size = y_pred.shape[0]
        
        _y = cp.Variable((1, batch_size))
        _G = cp.Parameter((2,1))
        G = y_pred.new(2,1)
        G[0,0] = 1. 
        G[1,0] = -1. 
        _h_x = cp.Parameter((2,batch_size))
        h_x = y_pred.new(2,batch_size)
        h_x[0,:] = constr_para[:, 1]
        h_x[1,:] = -constr_para[:, 0]
        _h_y = cp.Parameter((2,batch_size))
        h_y = y_pred.new(2,batch_size)
        h_y[0,:] = constr_para[:, 3]
        h_y[1,:] = -constr_para[:, 2]
        
        for i in range(6):
            _y_pred_i = cp.Parameter((1, batch_size))
            y_pred_i = y_pred.new(1,batch_size)
            y_pred_i[0,:] = y_pred[:, i]

            obj = cp.Minimize(0.5*cp.sum_squares(_y_pred_i-_y))
            cons = 0
            layer = 0
            if i < 3:
                cons = [_G@_y <= _h_x]
                prob = cp.Problem(obj, cons)
                layer = CvxpyLayer(prob, parameters=[_y_pred_i, _G, _h_x],
                        variables=[_y])

                no_except = True
                try:
                    y, = layer(y_pred_i, G, h_x)
                except:
                    print("not solvable")
                    no_except = False
            
                if no_except:
                    y_proj[:,i] = y[0,:]
                else:
                    y_proj[:,i] = y_pred_i[0,:]

            else:
                cons = [_G@_y <= _h_y]
                prob = cp.Problem(obj, cons)
                layer = CvxpyLayer(prob, parameters=[_y_pred_i, _G, _h_y],
                        variables=[_y])
                no_except = True
                try:
                    y, = layer(y_pred_i, G, h_y)
                except:
                    print("not solvable")
                    no_except = False
            
                if no_except:
                    y_proj[:,i] = y[0,:]
                else:
                    y_proj[:,i] = y_pred_i[0,:]

        return y_proj


