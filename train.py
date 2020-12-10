"""Use this python script for training.
"""

from options.opt_manager import OptManager
import data
from data.celeba_dataset import CelebaDataset
import optimization
import models
import utility.logging_fcts as logging_fcts

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import time
import os


if __name__ == '__main__':
    
    #option manager for parsing options
    opt_manager = OptManager(
            config_default = Path() / 'config_train.yaml',
            opt_def_default = Path() / 'opt_def_train.yaml',
            default_flow_style = False
            )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #parse options
    opts = opt_manager.parse()

    #save options specified by config file and command line arguments in a
    #final config file. Path for this file is opts.config_final
    opt_manager.obj2yaml(
            opts,
            dest=Path(opts.config_final)
            )

    #save the option definition file in the output directory
    opt_manager.opt_def2yaml(Path(opts.opt_def_final))

    #specify ID
    ID = opts.experiment_ID
    ckpt_file_str = 'ckpt_latest_' + ID

    #dataloader with dataset and further parameters specified in opts.
    dataloader = data.my_dataloader(opts)
    #dataloader for validation set
    dataloader_valid = data.my_dataloader(opts, sampler='valid')

    #select and load model from opts
    model = models.my_model(opts)
    #put model to device
    model.to(device)

    if opts.multiple_gpus:
        model = nn.DataParallel(model)
    
    model.train()

    #select loss fct from opts
    loss_fct = optimization.my_loss_fct(opts)

    #select optimizer from opts
    optimizer = optimization.my_optimizer(model.parameters(), opts)

    #create access to logging file
    logs = logging_fcts.Logs(Path(opts.log_file))

    start_time = time.time()

    i_batch_total = 0
    
    # Use this dictionary for storing information
    doc_dict = dict()
    # min_loss_valid_epoch: stores the minimum validation loss averaged over one
    # epoch 
    doc_dict['min_loss_valid_epoch'] = 999999
    # early_stopping_counter: a one is added when validation loss (averaged over
    # one epoch) increases in comparison to previous epoch, is set to 0 in
    # case it decreases. When counter is equal to opts.n_early_stop training is
    # stopped.
    doc_dict['early_stopping_counter'] = 0
    # loss_valid_total: for accumulating validation losses over one epoch
    doc_dict['loss_valid_total'] = 0.
    
    for i_epoch in np.arange(opts.start_epoch, opts.start_epoch + opts.n_epochs):
        start_time_epoch = time.time()
        start_time_data = time.time()
        iteration = 0
        doc_dict['loss_valid_total'] = 0
        count_valid = 0
        loss_train = 0.
        count_train = 0.
        start_time_load = time.time()
        for i_batch, data in enumerate(dataloader):
            end_time_data = time.time()
            time_data = (end_time_data - start_time_data) / opts.batch_size
            start_time_comp = end_time_data
            i_batch_total += 1

            y = data['y']
            #ship input/output to device
            y = y.to(device)
            y_pred = 0.
            if opts.model_module == 'constraintnet':
                x_img = data['img'].to(device)
                constr_para = data['constr_para'].to(device)
                y_pred = model(x_img, constr_para)
            else:
                x = data['img'].to(device)
                y_pred = model(x)                       

            iteration += y.shape[0]
            count_train += y.shape[0]

            loss = loss_fct(y_pred, y) * len(opts.lm_ordering_lm_order)
            loss_train += loss.item() * y.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            del loss 

            end_time_comp = time.time()
            #time for one forward and backward pass per data point
            time_comp = (end_time_comp - start_time_comp) / opts.batch_size

            #save logs with specified freq
            if i_batch_total % opts.log_freq == 0:
                loss_train_avg = loss_train / count_train
                loss_train = 0.
                count_train = 0
                logs.add_std_log(int(i_epoch), iteration,
                        log_type='train_loss',
                        time_data=time_data,
                        time_comp=time_comp,
                        **{opts.opts2loss_fct: loss_train_avg}
                        )

            #run evaluation on validation set
            if i_batch_total % opts.log_valid_freq == 0:
                model.eval()
                iteration_valid = 0
                running_loss_valid = 0.
                start_time_valid = time.time()
                with torch.no_grad():
                    for i_batch_valid, data_valid in enumerate(dataloader_valid):
                        if i_batch_valid + 1 > opts.n_batches_valid:
                            break
                        y_valid = data_valid['y']
                        y_valid = y_valid.to(device)
                        x_img_valid = data_valid['img'].to(device)
                        iteration_valid += x_img_valid.shape[0]
                        count_valid += x_img_valid.shape[0]

                        y_pred_valid = 0.
                        if opts.model_module == 'constraintnet':
                            constr_para_valid = data_valid['constr_para'].to(device)
                            y_pred_valid = model(x_img_valid, constr_para_valid)
                        else:
                            y_pred_valid = model(x_img_valid)

                        loss_valid = loss_fct(y_pred_valid, y_valid) * len(opts.lm_ordering_lm_order)
                        running_loss_valid += loss_valid.item() * x_img_valid.shape[0]

                end_time_valid = time.time()
                time_valid = end_time_valid - start_time_valid

                loss_valid = running_loss_valid / iteration_valid
                doc_dict['loss_valid_total'] += running_loss_valid

                logs.add_std_log(int(i_epoch), iteration,
                        log_type = 'valid_loss',
                        **{'time_valid': time_valid,
                            'size_valid_set': iteration_valid,
                            opts.opts2loss_fct: loss_valid}
                        )

                print('-> i_epoch: ', int(i_epoch), 'i_batch: ', i_batch,
                        'mseloss on validation set: ', loss_valid)
                model.train()

                
            if i_batch_total % opts.ckpt_freq == 0:
                # save latest model weights
                ckpt_file_latest = Path(opts.ckpt_dir) / Path(ckpt_file_str)
                torch.save(model.state_dict(), ckpt_file_latest)

            start_time_data = time.time()
            start_time_load = time.time()


        mean_valid_loss = doc_dict['loss_valid_total']
        if count_valid > 0:
            mean_valid_loss = doc_dict['loss_valid_total'] / count_valid

        #early stopping behavior
        if mean_valid_loss < doc_dict['min_loss_valid_epoch']:
            doc_dict['min_loss_valid_epoch'] = mean_valid_loss
            doc_dict['early_stopping_counter'] = 0
            ckpt_name = 'id_' + ID + '_ckpt_min_loss_epoch_{i_epoch}'.format(
                i_epoch=i_epoch)
            ckpt_file = Path(opts.ckpt_dir) / ckpt_name
            torch.save(model.state_dict(), ckpt_file)
        else:
            doc_dict['early_stopping_counter'] += 1

        if doc_dict['early_stopping_counter'] == opts.n_early_stop:
            print('Stopped training after {i_epoch} epochs.'.format(
                i_epoch=i_epoch
            ))
            break
