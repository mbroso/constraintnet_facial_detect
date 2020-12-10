"""This module implements a customized dataset loader with options and 
dataset selection according to opts.

For access to certain dataset a dataset class (subclass of 
torch.utils.data.Dataset) must be implemented in module named 
opts.dataset_module. In the same module a wrapper function with name 
opts.opts2dataset must be implemented which creates an instance of the dataset
class based on the parsed options. The module file must be located in the data 
directory.
"""

import importlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def my_dataset(opts):
    """Creates dataset object according options in opts.

    Calls dataset wrapper function opts.opts2dataset in module 
    opts.dataset_module to create dataset instance.

    Args:
        opts (obj): Namespace object with options, required 
            attributes are opts.dataset_module and opts.opts2dataset.
    
    Returns:
        my_dataset (obj): Instantiated dataset object constructed by dataset 
            wrapper opts.opts2datset in module opts.dataset_module.
    """
    dataset_import = 'data.' + opts.dataset_module
    dataset_lib = importlib.import_module(dataset_import)

    my_dataset = None
    for name, fct in dataset_lib.__dict__.items():
        if name==opts.opts2dataset:
            my_dataset = fct(opts) 

    if my_dataset==None:
        raise NotImplementedError(
                """In {dataset_import} is no function {opts2datset} 
                implemented.""".format(
                    dataset_import=dataset_import, 
                    opts2dataset=opts.opts2dataset
                    )
                )

    return my_dataset


def my_dataloader(opts, sampler=None):
    """Creates customized dataloader with dataset selection and 
    parametrization according to opts.

    Args:
        opts (obj): Namespace object with options.
        sampler (str): Specify train, valid or test to overwrite opts.sampler.

    Returns:
        my_dataloader (obj): Customized dataloader. Dataloader of PyTorch 
            instantiated with choosen dataset and options.
    """

    dataset = my_dataset(opts)
    print("""Dataset access was established by calling dataset wrapper function 
            {opts2dataset} defined in module {dataset_module}.""".format(
        dataset_module=opts.dataset_module,
        opts2dataset=opts.opts2dataset
        ))
    split = dataset.split()
    
    if sampler is None:
        sampler = opts.sampler

    if opts.sampler_rand:
        sampler_range = dataset.get_sampler(split[sampler])
    elif not opts.sampler_rand:
        sampler_range = dataset.get_fixed_sampler(split[sampler])
    else:
        TypeError('No sampler type specified')

    my_dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            num_workers=opts.threads,
            sampler = sampler_range,
            pin_memory=True
            )

    return my_dataloader
