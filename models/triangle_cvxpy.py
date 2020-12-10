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


def opts2triangle_projection(opts):
    """Creates TriangleProjection nn.Modules by calling its constructor with
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        TriangleProjection (obj): Instantiated TriangleProjection nn.Module.
    """

    return TriangleProjection()

class TriangleProjection(nn.Module):
    """This is a functor which performs the projection for triangle
    constraints. The package cvxpy is used.
    """
    def __init__(self):
        super(TriangleProjection, self).__init__()

    def __call__(self, y_pred, constr_para):
        """Projects y_pred on region defined by constr_para.

        Args:
            y_pred (obj): Pytorch tensor with shape (batch_size, 2). And the
                second dimension is for the landmark positions: nose_x,
                nose_y.
            constr_para (obj): Pytorch tensor with shape (batch_size, 6). And
                the second dimension is for the bounding box boundaries l_x,
                u_x, l_y, u_y.

        Returns:
            y_proj (obj): Pytorch tensor with projected version of y_pred.
        """
        y_proj = y_pred.new(y_pred.shape)
        batch_size = y_pred.shape[0]

        for i in range(batch_size):
            y_pred_i = y_pred.new(2,)
            y_pred_i[:] = y_pred[i,:]
            x_0 = constr_para[i, 0]
            y_0 = constr_para[i, 1]
            x_1 = constr_para[i, 2]
            y_1 = constr_para[i, 3]
            x_2 = constr_para[i, 4]
            y_2 = constr_para[i, 5]
            n_1_x = (y_0 - y_2)
            n_1_y = -(x_0 - x_2)
            n_2_x = (y_1 - y_0)
            n_2_y = -(x_1 - x_0)
            n_3_x = (y_2 - y_1)
            n_3_y = -(x_2 - x_1)

            G_1 = y_pred.new(1,2)
            G_1[0,0] = n_1_x
            G_1[0,1] = n_1_y
            h_1 = y_pred.new(1,)
            h_1[0] = n_1_x*x_0 + n_1_y*y_0
            G_2 = y_pred.new(1,2)
            G_2[0,0] = n_2_x
            G_2[0,1] = n_2_y
            h_2 = y_pred.new(1,)
            h_2[0] = n_2_x*x_1 + n_2_y*y_1
            G_3 = y_pred.new(1,2)
            G_3[0,0] = n_3_x
            G_3[0,1] = n_3_y
            h_3 = y_pred.new(1,)
            h_3[0] = n_3_x*x_2 + n_3_y*y_2

            _y_pred_i = cp.Parameter((2,))
            _y = cp.Variable((2,))
            _G_1 = cp.Parameter((1,2))
            _h_1 = cp.Parameter((1,))
            _G_2 = cp.Parameter((1,2))
            _h_2 = cp.Parameter((1,))
            _G_3 = cp.Parameter((1,2))
            _h_3 = cp.Parameter((1,))
            obj = cp.Minimize(0.5*cp.sum_squares(_y_pred_i-_y))
            cons = [_G_1@_y<= _h_1, _G_2@_y<= _h_2, _G_3@_y<= _h_3]
            prob = cp.Problem(obj, cons)
            layer = CvxpyLayer(prob, parameters=[_y_pred_i, _G_1, _h_1, _G_2,
                _h_2, _G_3, _h_3], variables=[_y])

            no_except = True
            try:
                y, = layer(y_pred_i, G_1, h_1, G_2, h_2, G_3, h_3)
            except:
                print("not solvable")
                no_except = False
        
            if no_except:
                y_proj[i,:] = y[:]
            else:
                y_proj[i,:] = y_pred_i[:]


        return y_proj

