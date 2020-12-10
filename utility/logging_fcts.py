"""This module allows reading and writing log files.

The log files are yaml files that store logs as items of a list. 
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time
import matplotlib.ticker as ticker

def opts2logs(opts):
    """Function to create Logs object from options.

    Args:
        opts (obj): Namespace object with options. Required option is
            opts.log_file.
    """
    return Logs(Path(opts.log_file))

class Logs:
    """This class provides functionality for writing and reading logs to/from 
    file.

    Logs are saved as a yaml file. The yaml file consists of a list with log 
    items.
    """
    def __init__(self, log_file):
        """Initialization creates files and saves location.

        Args:
            log_file (obj): Path (Path object of pathlib) for saving log file.   
        """
        self.log_file = log_file
        self.create_log_file()

    def create_log_file(self):
        """Creates an empty yaml logfile.
        """
        with self.log_file.open('w') as f:
            pass
    
    def read_logs(self):
        """Read the log file and return the content.

        Returns:
            logs (list): List of log items.
        """
        with self.log_file.open('r') as f:
            logs = yaml.safe_load(f)
        
        return logs

    def write_logs(self, logs):
        """Writes (overwrites) logs to log file.

        Args:
            logs (obj): Object that is supported by yaml.
        """
        with self.log_file.open('w') as f:
            yaml.safe_dump(logs, f)

    def add_log(self, log):
        """Add a log to the log file.

        Args:
            logs (obj): Object that is supported by yaml.
        """
        logs = self.read_logs()
        if logs is None:
            logs = [log,]
        else:
            logs.append(log)
        self.write_logs(logs)
           
    def add_std_log(self, epoch, iteration, log_type='std_log', time_data=None, 
            time_comp=None, **log_dict):
        """Adds a log with certain format to the log file.

        Format of log:
            type: 'std_log'
            epoch: xxx
            iteration: xxx
                ...
            time_data: xxx
            time_comp: xxx
            additional entries from log_dict

        Args:
            log_type (str): Type of the log. Here std_log.
            epoch (int): Index of current epoch.
            iteration (int): Number of passed samples within current epoch.
            time_data: Time for loading and preprocessing mini batch.
            time_comp: Time for forward and backward pass of mini batch.
            time_comp: Time information for log.
            **log_dict: Dictionary with keyword arguments as additional free 
                information. E.g. losses as keyword arguments with keyword
                encoding their type. 

        """

        log = {
                'log_type': log_type, 
                'epoch': epoch, 
                'iteration': iteration,
                }

        for log_type, log_value in log_dict.items():
            log[log_type] = log_value
        
        if not time_data is None:
            log['time_data'] = time_data

        if not time_comp is None:
            log['time_comp'] = time_comp

        self.add_log(log)
    
    
