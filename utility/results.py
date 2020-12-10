"""This module allows to save and print the result on test set.
"""

import yaml
import pandas as pd
from pathlib import Path
import numpy as np
import datetime

class Results:
    """This class saves results on test set as yaml file and allows to print
    results.
    """
    def __init__(self, results_file=Path('./results.yaml'), comment=None, 
            size_test_set=None, time_test=None, **eval_metrics):
        """Initialize Result by passing evaluation metrics with their values as
        keyword arguments.

        Args:
            results_file (obj): Destination for saving results file (Path 
                object of pathlib).
            comment (str): Some comment, e.g. dataset name, to better 
                understand the file later.
            size_test_set (int). Number of elements in the test set.
            time_test (float): Time in seconds for testing.
            **eval_metrics (float): Variable number of keyword arguments for 
                evaluation metrics.
        """
        self.results_file = results_file
        #dictionary for saving result description
        self.results = {'header': 'Results on test set.'}
        self.eval_metrics = eval_metrics
        if not comment is None:
            self.results['comment'] = comment
        if not size_test_set is None:
            self.results['size_test_set'] = size_test_set
        if not time_test is None:
            self.time_test = self.results['time_test'] = time_test
        self.results['eval_metrics'] = {}
        for eval_metric, value in self.eval_metrics.items():
            self.results['eval_metrics'][eval_metric] = value
    
    def add_eval_metrics(**eval_metrics):
        """Add arbitrary number of evaluation metrics.

        Args:
            **eval_metrics (float): Variable number of keyword arguments for 
                evaluation metrics.
        """
        for eval_metric, value in eval_metrics.items():
            self.results['eval_metrics'][eval_metric] = value

    def write(self, dest=None):
        """Writes results to yaml file.

        Args:
            dest (obj): Path for saving yaml file (Path object of pathlib). If
                not specified self.results_file is used.
        """
        if dest is None:
            dest = self.results_file
        with dest.open('w') as f:
            yaml.safe_dump(self.results, f)
    
    def __str__(self):
        """Prints results information well-aranged.
        """
        print(self.results)
        message = '-----------------------------' + '\n'
        message += self.results['header'] + '\n'
        message += 'Number of data points in test set: ' 
        message += str(self.results['size_test_set']) + '\n'
        message += 'Time for test run [h:mm::ss]: ' 
        time_test = str(datetime.timedelta(seconds=self.results['time_test']))
        message += time_test + '\n'
        message += 'Evaluation metrics: ' + '\n'
        for eval_metric, value in self.eval_metrics.items():
            message += '    ' + eval_metric + ': ' + str(value) + '\n'
        message += '-----------------------------'
        return message



