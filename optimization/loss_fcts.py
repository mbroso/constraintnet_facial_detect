"""This module allows to instantiate loss objects based on parsed options.

This module comprises functions to instantiate loss objects. For chosing a loss
object, one of these functions can be selected via the option
opts.opts2loss_fct, opts.opts2loss_fct must match the function name, e.g.
opts.opts2loss_fct = 'opts2mseloss' choses the function to instantiate
a mean squared error loss object. The functions in this module must be callable
with the options object (opts).
"""

import torch.nn as nn

def opts2mseloss(*opts):
    """Instantiates mean square error loss object according to parsed options.

    MSEloss does not take options. Therefore this argument ist just to be
    align with interface format.

    Args:
        *opts (None): Placeholder, since no options are acutally needed.

    Returns:
        loss_module (obj): Instantiated MSEloss module.
    """
    return nn.MSELoss()
