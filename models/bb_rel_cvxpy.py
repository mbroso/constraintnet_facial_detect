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


def opts2bb_rel_projection(opts):
    """Creates BbRelProjection nn.Modules by calling its constructor with
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        BbRelProjection (obj): Instantiated BbRelProjection nn.Module.
    """

    return BbRelProjection()

class BbRelProjection(nn.Module):
    """This is a functor which performs the projection for bounding box
    and additional relative constraints. The package cvxpy is used.
    """
    def __init__(self):
        super(BbRelProjection, self).__init__()

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
        
        #nose_x part (interval in 1d)       
        _y_1 = cp.Variable((1, batch_size))
        _y_pred_1 =  cp.Parameter((1, batch_size))
        _G_1 =cp.Parameter((2, 1))
        _h_1 = cp.Parameter((2, batch_size))

        obj_1 = cp.Minimize(0.5*cp.sum_squares(_y_pred_1-_y_1))
        cons_1 = [_G_1@_y_1 <= _h_1]
        prob_1 = cp.Problem(obj_1, cons_1)
        layer_1 = CvxpyLayer(prob_1, parameters=[_y_pred_1, _G_1, _h_1], variables=[_y_1])
        
        G_1 = y_pred.new(2,1)
        G_1[0,0] = 1.
        G_1[1,0] = -1.
        h_1 = y_pred.new(2, batch_size) 
        h_1[0,:] = constr_para[:, 1] #u_x
        h_1[1,:] = -constr_para[:, 0] #-l_x
        y_pred_1 = y_pred.new(1, batch_size)
        y_pred_1[0,:] = y_pred[:, 0]
        
        no_except = True
        try:
            y_1, = layer_1(y_pred_1, G_1, h_1)
        except:
            print("not solvable")
            no_except = False
        
        if no_except:
            y_proj[:,0] = y_1[0, :]
        else:
            y_proj[:,0] = y_pred_1[0, :]

        #lefteye_x, righteye_x part (triangle in 2d) 
        _y_2 = cp.Variable((2, batch_size))
        _y_pred_2 =  cp.Parameter((2, batch_size))
        _G_2_1 =cp.Parameter((1, 2))
        _h_2_1 = cp.Parameter((1, batch_size))
        _G_2_2 =cp.Parameter((1, 2))
        _h_2_2 = cp.Parameter((1, batch_size))
        _G_2_3 =cp.Parameter((1, 2))
        _h_2_3 = cp.Parameter((1, batch_size))

        obj_2 = cp.Minimize(0.5*cp.sum_squares(_y_pred_2-_y_2))
        cons_2 = [_G_2_1@_y_2<= _h_2_1, _G_2_2@_y_2<= _h_2_2, _G_2_3@_y_2<= _h_2_3]
        prob_2 = cp.Problem(obj_2, cons_2)
        layer_2 = CvxpyLayer(prob_2, parameters=[_y_pred_2, _G_2_1, _h_2_1,
            _G_2_2, _h_2_2, _G_2_3, _h_2_3], variables=[_y_2])
        
        G_2_1 = y_pred.new(1,2)
        G_2_1[0,0] = -1.
        G_2_1[0,1] = 0.
        h_2_1 = y_pred.new(1, batch_size)
        h_2_1[0,:] = -constr_para[:, 0] #-l_x
        G_2_2 = y_pred.new(1,2)
        G_2_2[0,0] = 0.
        G_2_2[0,1] = 1.
        h_2_2 = y_pred.new(1, batch_size)
        h_2_2[0,:] = constr_para[:, 1] #u_x
        G_2_3 = y_pred.new(1,2)
        G_2_3[0,0] = 1.
        G_2_3[0,1] = -1.
        h_2_3 = y_pred.new(1, batch_size)
        h_2_3[0,:] = 0
        y_pred_2 = y_pred.new(2, batch_size)
        y_pred_2[0,:] = y_pred[:, 1]
        y_pred_2[1,:] = y_pred[:, 2]

        no_except = True
        try:
            y_2, = layer_2(y_pred_2, G_2_1, h_2_1, G_2_2, h_2_2, G_2_3, h_2_3)
        except:
            print("not solvable")
            no_except = False
        
        if no_except:
            y_proj[:,1] = y_2[0, :]
            y_proj[:,2] = y_2[1, :]
        else:
            y_proj[:,1] = y_pred_2[0, :]
            y_proj[:,2] = y_pred_2[1, :]


        #lefteye_y, righteye_y, nose_y part (pyramid in 3d)
        _y_3 = cp.Variable((3, batch_size))
        _y_pred_3 =  cp.Parameter((3, batch_size))
        _G_3_1 = cp.Parameter((1, 3))
        _h_3_1 = cp.Parameter((1, batch_size))
        _G_3_2 = cp.Parameter((1, 3))
        _h_3_2 = cp.Parameter((1, batch_size))
        _G_3_3 = cp.Parameter((1, 3))
        _h_3_3 = cp.Parameter((1, batch_size))
        _G_3_4 = cp.Parameter((1, 3))
        _h_3_4 = cp.Parameter((1, batch_size))
        _G_3_5 = cp.Parameter((1, 3))
        _h_3_5 = cp.Parameter((1, batch_size))


        obj_3 = cp.Minimize(0.5*cp.sum_squares(_y_pred_3-_y_3))
        cons_3 = [_G_3_1@_y_3<= _h_3_1, _G_3_2@_y_3<= _h_3_2, _G_3_3@_y_3<= _h_3_3,
                _G_3_4@_y_3<= _h_3_4, _G_3_5@_y_3<= _h_3_5]
        prob_3 = cp.Problem(obj_3, cons_3)
        layer_3 = CvxpyLayer(prob_3, parameters=[_y_pred_3, _G_3_1, _h_3_1,
            _G_3_2, _h_3_2, _G_3_3, _h_3_3, _G_3_4, _h_3_4, _G_3_5, _h_3_5],
            variables=[_y_3])
        
        #righteye_y >= l_y
        G_3_1 = y_pred.new(1,3)
        G_3_1[0,0] = 0.
        G_3_1[0,1] = -1.
        G_3_1[0,2] = 0.
        h_3_1 = y_pred.new(1, batch_size)
        h_3_1[0,:] = -constr_para[:, 2]

        #lefteye_y >= l_y
        G_3_2 = y_pred.new(1,3)
        G_3_2[0,0] = -1.
        G_3_2[0,1] = 0.
        G_3_2[0,2] = 0.
        h_3_2 = y_pred.new(1, batch_size)
        h_3_2[0,:] = -constr_para[:, 2]

        #nose_y <= u_y
        G_3_3 = y_pred.new(1,3)
        G_3_3[0,0] = 0.
        G_3_3[0,1] = 0.
        G_3_3[0,2] = 1.
        h_3_3 = y_pred.new(1, batch_size)
        h_3_3[0,:] = constr_para[:, 3]

        #lefteye_y <= y_nose
        G_3_4 = y_pred.new(1,3)
        G_3_4[0,0] = 1.
        G_3_4[0,1] = 0.
        G_3_4[0,2] = -1.
        h_3_4 = y_pred.new(1, batch_size)
        h_3_4[0,:] = 0.

        #righteye_y <= y_nose
        G_3_5 = y_pred.new(1,3)
        G_3_5[0,0] = 0.
        G_3_5[0,1] = 1.
        G_3_5[0,2] = -1.
        h_3_5 = y_pred.new(1, batch_size)
        h_3_5[0,:] = 0.

        y_pred_3 = y_pred.new(3, batch_size)
        y_pred_3[0,:] = y_pred[:, 3]
        y_pred_3[1,:] = y_pred[:, 4]
        y_pred_3[2,:] = y_pred[:, 5]

        no_except = True
        try:
            y_3, = layer_3(y_pred_3, G_3_1, h_3_1, G_3_2, h_3_2, G_3_3, h_3_3, G_3_4,
                h_3_4, G_3_5, h_3_5)
        except:
            print("not solvable")
            no_except = False
        
        if no_except:
            y_proj[:,3] = y_3[0, :]
            y_proj[:,4] = y_3[1, :]
            y_proj[:,5] = y_3[2, :]
        else:
            y_proj[:,3] = y_pred_3[0, :]
            y_proj[:,4] = y_pred_3[1, :]
            y_proj[:,5] = y_pred_3[2, :]
        
        return y_proj


