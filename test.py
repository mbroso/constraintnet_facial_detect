"""Use this python script for inference.

For specifying the GPU type:
    CUDA_VISIBLE_DEVICES = <GPU_IDX> python test.py 
"""

from options.opt_manager import OptManager
import data
from data.celeba_dataset import CelebaDataset
import models
import optimization
import utility.results as results

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import time

np.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':

    #option manager for parsing options
    opt_manager = OptManager(
            config_default = Path() / 'config_test.yaml',
            opt_def_default = Path() / 'opt_def_test.yaml',
            default_flow_style = False
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Parse options
    opts = opt_manager.parse()

    # save options specified by config file and command line arguments in a
    # final config file. Path for this file is opts.config_final.
    opt_manager.obj2yaml(
        opts,
        dest=Path(opts.config_final)
    )

    # save the option definition file in the output directory
    opt_manager.opt_def2yaml(Path(opts.opt_def_final))
 
    # flag for evaluating lefteye righteye landmarks
    flag_lms = False
    if len(opts.lm_ordering_lm_order) > 2:
        flag_lms = True

    dataloader = data.my_dataloader(opts)

    #load model from opts
    model = models.my_model(opts)
    model.to(device) 

    if opts.multiple_gpus:
        model = nn.DataParallel(model)
    
    print('Reload from checkpoint {ckpt_file}.'.format(
        ckpt_file=opts.reload_ckpt_file))
    ckpt = torch.load(Path(opts.reload_ckpt_file))
    model.load_state_dict(ckpt)

    #evaluation mode
    model.eval()

    loss_fct = optimization.my_loss_fct(opts)

    
    mse_loss = []
    mse_nose_x = []
    mse_nose_y = []
    times = []
    if flag_lms:
        mse_lefteye_x = []
        mse_righteye_x = []
        mse_lefteye_y = []
        mse_righteye_y = []

    for i in np.arange(30):
        running_loss = 0.
        running_loss_lms = len(opts.lm_ordering_lm_order) * [0,]
        iteration = 0
        with torch.no_grad():
            start_time = time.time()
            for i_batch, data in enumerate(dataloader):
                y = data['y']
                y = y.to(device)
                x_img = data['img'].to(device)

                y_pred = 0.
                if opts.model_module == 'constraintnet':
                    constr_para = data['constr_para'].to(device)
                    y_pred = model(x_img, constr_para)

                else:
                    y_pred = model(x_img)

                iteration += x_img.shape[0]
                
                loss = loss_fct(y_pred, y) * len(opts.lm_ordering_lm_order)
                running_loss += loss.item() * x_img.shape[0]

                if i_batch % 20 == 0:
                    print('Test loss: {loss}, progress: {progress} %'.format(
                        loss=loss,
                        progress=round(iteration / 19961, 2)*100)
                        )
                for i in range(len(opts.lm_ordering_lm_order)):
                    loss_lm = loss_fct(y_pred[:, i], y[:, i])
                    running_loss_lms[i] += loss_lm.item() * x_img.shape[0]

        
        #save losses in dictionary eval_metrics
        loss = running_loss / iteration
        eval_metrics = {'mse_loss': loss}
        for idx, lm in enumerate(opts.lm_ordering_lm_order):
            key = 'mse_' + str(lm)
            loss_lm = running_loss_lms[idx] / iteration
            eval_metrics[key] = loss_lm 

        end_time = time.time()
        time_test = end_time - start_time
        eval_metrics['times'] = time_test

        test_results = results.Results(results_file=Path(opts.results_file),
                                       comment='test results',
                                       size_test_set=iteration,
                                       time_test=time_test,
                                       **eval_metrics
                                       )
        test_results.write()
        print(test_results)
   
        mse_loss.append(eval_metrics['mse_loss'])
        mse_nose_x.append(eval_metrics['mse_nose_x'])
        mse_nose_y.append(eval_metrics['mse_nose_y'])
        times.append(eval_metrics['times'])
        if flag_lms:
            mse_lefteye_x.append(eval_metrics['mse_lefteye_x'])
            mse_righteye_x.append(eval_metrics['mse_righteye_x'])
            mse_lefteye_y.append(eval_metrics['mse_lefteye_y'])
            mse_righteye_y.append(eval_metrics['mse_righteye_y'])
    
    eval_metrics = {'mse_loss': float(np.array(mse_loss).mean()) }
    eval_metrics['times'] = float(np.array(times).mean())
    eval_metrics['mse_nose_y'] = float(np.array(mse_nose_y).mean())
    eval_metrics['mse_nose_x'] = float(np.array(mse_nose_x).mean())
    if flag_lms:
        eval_metrics['mse_lefteye_x'] = float(np.array(mse_lefteye_x).mean())
        eval_metrics['mse_righteye_x'] = float(np.array(mse_righteye_x).mean())
        eval_metrics['mse_lefteye_y'] = float(np.array(mse_lefteye_y).mean())
        eval_metrics['mse_righteye_y'] = float(np.array(mse_righteye_y).mean())
    
    eval_metrics['mse_loss_std'] = float(np.array(mse_loss).std())
    eval_metrics['mse_nose_x_std'] = float(np.array(mse_nose_x).std())
    eval_metrics['mse_nose_y_std'] = float(np.array(mse_nose_y).std())
    eval_metrics['times_std'] = float(np.array(times).std())
    if flag_lms:
        eval_metrics['mse_lefteye_x_std'] = float(np.array(mse_lefteye_x).std())
        eval_metrics['mse_righteye_x_std'] = float(np.array(mse_righteye_x).std())
        eval_metrics['mse_lefteye_y_std'] = float(np.array(mse_lefteye_y).std())
        eval_metrics['mse_righteye_y_std'] = float(np.array(mse_righteye_y).std())



    test_results = results.Results(results_file=Path(opts.results_file),
            comment='test results over 30 test runs',
            size_test_set=iteration,
            time_test=time_test,
            **eval_metrics)
    test_results.write()
    print(test_results)


