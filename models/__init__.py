"""This module implements a model selection by filename and class name 
according to opts.

Opts must contain attributes opts.model_module and opts.model_cls. The 
specified class must be a subclass of nn.Module. 
"""

import torch.nn as nn
import importlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def my_model(opts):
    """Creates model object according to settings in parsed options.

    Calls function with name opts.opts2model in module opts.model_module to
    create model instance.

    Args:
        opts (obj): Namespace object with options. Required attributes are
            opts.model_module and opts.opts2model.
    
    Returns:
        my_model (obj): Instantiated model object construtcted by function 
            opts.opts2model in module opts.model_module.

    Raises:
        NotImplementedError: When model wrapper opts.opts2model does not exist 
            in model module opts.model_module.
    """

    model_import = 'models.' + opts.model_module
    model_lib = importlib.import_module(model_import)
    
    my_model = None
    for name, fct in model_lib.__dict__.items():
        if name==opts.opts2model:
            my_model = fct(opts)

    if my_model==None:
        raise NotImplementedError(
                """Model wrapper function {opts2model} is not implemented in 
                model module {model_module}""".format(
                    opts2model=opts.opts2model, 
                    model_module=opts.model_module
                    )
                )
    
    print("""Model was constructed by calling model function 
            {opts2model} in model module {model_module}.""".format(
            opts2model=opts.opts2model,
            model_module=opts.model_module
            )
        )
    

    return my_model
