from parse import * 
from data.tpch_utils import PSQLTPCHDataSet
from model.QPPnets import QPPNet_t, QPPNet_m
import torch
import numpy as np
from utils import *
import random

def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    opt = parser.parse_args()
    opt.type = 'memory'

    set_seed()
    if opt.dataset == "PSQLTPCH":
        dataset = PSQLTPCHDataSet(opt)
    print("dataset_size", dataset.datasize)
    
    torch.set_default_tensor_type(torch.FloatTensor)
    logf = open(opt.type+'-'+opt.logfile, 'w+')
    save_opt(opt, logf)
    total_iter = 0

    if opt.type == 'latency':
        qpp = QPPNet_t(opt)
    else:
        qpp = QPPNet_m(opt)
    qpp.test_dataset = dataset.test_dataset

    for epoch in range(opt.start_epoch, opt.end_epoch):
        epoch_iter = 0     # the number of training iterations in current epoch, reset to 0 every epoch

        samp_dicts_list = dataset.sample_data_list()
        for samp_dicts in samp_dicts_list:
            if not samp_dicts: continue
            qpp.set_input(samp_dicts)
            qpp.optimize_parameters(epoch)  # model training

        qpp.validation(epoch)
        total_iter += dataset.datasize
        train_info_str = f"epoch: {epoch}; iter_num: {total_iter}; total_loss: {qpp.last_total_loss}; " + \
                         f"test_loss:{qpp.test_loss_value}; mean_abs_diff: {qpp.last_mean_abs_diff}; " + \
                         f"mean_abs_relate_err: {qpp.last_mean_abs_relate_er}; R(q): {qpp.last_rq}"
        
        logf.write(train_info_str)
        losses = qpp.get_current_losses()
        loss_str = "losses: "
        for op in losses:
            loss_str += str(op) + " [" + str(losses[op]) + "]; "
        logf.write(loss_str + '\n')

    
        print(train_info_str)
        print(loss_str)

        if (epoch + 1) % opt.save_latest_epoch_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print(f'saving the latest model (epoch {epoch + 1}, total_iters {total_iter})' )
            qpp.save_units(epoch + 1)

    logf.close()
