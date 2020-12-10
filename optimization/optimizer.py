"""This module allows to instantiate optimizer objects based on parsed options.

This module comprises functions to instantiate optimizer objects. For chosing an
optimizer object, one of these functions can be selected via the option
opts.opts2optimizer, e.g. opts2optimizer = 'opts2adam' choses the function to
instantiate Adam optimizer. The functions of this module must be callable with
an parameter object (params, contains the parameters of the neural network) and
the options object (opts).   
"""

import torch.optim as optim

def opts2adam(params, opts):
    """Instantiates Adam optimizer according to parsed options.

    Args:
        params (obj): The parameters of the neural network.
        opts (None): Namespace object with options for adam optimizer. Required
            attributes are opts.learning_rate, opts.beta_first_moment,
            opts.beta_second_moment, opts.eps, opts.weight_decay, opts.amsgrad.

    Returns:
        optimizer (obj): Instantiated Adam optimization object.
    """

    lr = opts.learning_rate
    #beta of exponential moving average of first moment of gradient
    #this implements gradient momemtum
    beta_first_moment = opts.beta_first_moment
    #beta of exponential moving average of second moment of gradient
    #this influences the individual learning rate of the parameters depending
    #on their magnitude
    beta_second_moment = opts.beta_second_moment
    #epsilon added to the denominator to stabilize it  
    eps = opts.eps
    #weight decay (L2 penalty) 
    weight_decay = opts.weight_decay
    #whether to use the AMSgrad variant of this algorithm
    amsgrad = opts.amsgrad

    #instantiate adam optimization module
    adam = optim.Adam(params, lr, (beta_first_moment, beta_second_moment), 
            weight_decay, amsgrad)

    return adam
