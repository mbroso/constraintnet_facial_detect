"""This module is an interface to select loss functions and optimizer according
to parsed options.

For chosing a certain loss object, a corresponding function must be defined in
the module optimization.loss_fcts. The function my_loss_fct in this file selects
a function in optimization.loss_fcts via the option opts.opts2loss_fct. For
chosing a certain optimizer object, a corresponding function must be defined in
the module optimization.optimizer. The function my_optimizer in this file
selects a function in optimization.optimizer via the option opts.opts2optimizer.
"""

import torch.nn as nn
import importlib
from pathlib import Path


def my_loss_fct(opts):
    """Selects and instantiates loss object according to parsed options.

    Selects function from the module optimization.loss_fcts with name
    opts.opts2loss_fct to instantiate loss object.

    Args:
        opts (obj): Namespace object with options. Required attribute is
            attribute opts.opts2loss_fct.
    
    Returns:
        my_loss (obj): Instantiated loss object.

    Raises:
        NotImplementedError: When no function with name opts.opts2loss_fct
            exists in module optimization.loss_fcts.
    """

       
    loss_fcts_import = 'optimization.loss_fcts'
    loss_fcts_lib = importlib.import_module(loss_fcts_import)
    
    #Pick loss function wrapper out of optimization.loss_fcts
    my_loss = None
    for name, fct in loss_fcts_lib.__dict__.items():
        if name == opts.opts2loss_fct:
            my_loss = fct(opts)

    if my_loss==None:
        raise NotImplementedError(
                """In {loss_fcts_import} is no function with name 
                {opts2loss_fct} implemented.""".format(
                    loss_fcts_import=loss_fcts_import, 
                    opts2loss_fct=opts.opts2loss_fct
                    )
                )

    print("""Loss object instantiated via function with name
            {opts2loss_fct} from module {loss_fcts_import}.""".format(
                opts2loss_fct=opts.opts2loss_fct,
                loss_fcts_import=loss_fcts_import
                )
            )
     

    return my_loss



def my_optimizer(params, opts):
    """Selects and instantiates optimizer object according to parsed options.

    Selects function from the module optimization.optimizer with name
    opts.opts2optimzer to instantiate optimizer object.

    Args:
        opts (obj): Namespace object with options. Required attribute is
            attribute opts.opts2optimizer.
    
    Returns:
        my_optimizer (obj): Instantiated optimizer object.

    Raises:
        NotImplementedError: When no function with name opts.opts2optimizer
            exists in module optimization.optimizer.

    """

    optimizer_import = 'optimization.optimizer'
    optimizer_lib = importlib.import_module(optimizer_import)
    
    #Pick function to create optimizer object from module
    #optimization.optimizer
    my_optimizer = None
    for name, fct in optimizer_lib.__dict__.items():
        if name == opts.opts2optimizer:
            my_optimizer = fct(params, opts)

    if my_optimizer==None:
        raise NotImplementedError(
                """In {optimizer_import} is no function with name 
                {opts2optimizer} implemented.""".format(
                    optimizer_import=optimizer_import, 
                    opts2optimizer=opts.opts2optimizer
                    )
                )
    
    print("""Optimizer object instantiated via function with name
            {opts2optimizer} from module 
            {optimizer_import}.""".format(
                opts2optimizer=opts.opts2optimizer,
                optimizer_import=optimizer_import 
                )
            )
     
    return my_optimizer



