import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

import functools, os
import numpy as np
import json

from model.metric import Metric

basic = 3
# TPCH
from data.tpch_utils import tpch_dim_dict

# For computing loss
def squared_diff(output, target):
    return torch.sum((output - target)**2)

###############################################################################
#                        Operator Neural Unit Architecture                    #
###############################################################################
# Neural Unit that covers all operators
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, dim_dict, num_layers=5, hidden_size=128,
                 output_size=32, norm_enabled=False):
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim = dim_dict[node_type])
    def build_block(self, num_layers, hidden_size, output_size, input_dim):

        assert(num_layers >= 2)
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()] # input layer
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()] # hidden layer
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()] # output layer

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        out = self.dense_block(x)
        return out

###############################################################################
#                               QPP Net Architecture                          #
###############################################################################

class QPPNet():
    def __init__(self, opt):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                                             else torch.device('cpu:0')
        self.save_dir = opt.save_dir
        self.test = False
        self.test_time = opt.test_time
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset

        if opt.dataset == "PSQLTPCH":
            self.dim_dict = tpch_dim_dict

        self.opt_obj = list()

        self.last_total_loss = None
        self.last_mean_abs_relate_er = None
        self.mean_abs_relate_err = None
        self.rq = 0
        self.last_rq = 0
        self.aim_str = 'total_time'

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.units = {}
        self.optimizers = {}
        self.schedulers = {} 
        self.best = 100000
        for operator in self.dim_dict:
            # 初始化网络
            self.units[operator] = NeuralUnit(operator, self.dim_dict).to(self.device)
            if opt.SGD: # 设置优化器
                optimizer = torch.optim.SGD(self.units[operator].parameters(), lr=opt.lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(self.units[operator].parameters(), opt.lr) #opt.lr
            if opt.scheduler: # 学习率调整
                sc = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,gamma=opt.gamma)
                self.schedulers[operator] = sc
            self.optimizers[operator] = optimizer

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [self.dummy] for operator in self.opt_obj}

        self.curr_batch_losses_dict = {operator: 0 for operator in self.opt_obj}
        self.batch_total_loss_value = None
        self._test_losses = dict()

        # 导入新模型
        if opt.start_epoch > 0:
            self.load(opt.start_epoch)

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def _forward(self, epoch):
        # self.input is a list of preprocessed plan_vec_dict
        batch_total_loss_value = torch.zeros(1).to(self.device)
        batch_total_losses_dict = {operator: [torch.zeros(1).to(self.device)] \
                        for operator in self.opt_obj}
        
        if self.test:
            abs_diff = []
            abs_relate_err_list = []

        all_real_value, all_pred_value = None, None

        data_size = 0
        total_mean_mae = torch.zeros(1).to(self.device)

        for idx, samp_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.opt_obj}

            _, pred_v= self._forward_oneQ_batch(samp_dict)

            epsilon = torch.finfo(pred_v.dtype).eps

            data_size += len(samp_dict['total_time'])
            # 计算该query的损失函数
            D_size = 0
            subbatch_loss_value = torch.zeros(1).to(self.device)
            for operator in self.acc_loss:
                acc_loss_cat = torch.cat(self.acc_loss[operator])
                D_size += acc_loss_cat.shape[0]
                subbatch_loss_value += torch.sum(acc_loss_cat)
                batch_total_losses_dict[operator].append(acc_loss_cat)
            subbatch_loss_value = torch.mean(torch.sqrt(subbatch_loss_value / D_size))
            batch_total_loss_value += subbatch_loss_value * samp_dict['subbatch_size']


            if self.test:
                real_v = torch.from_numpy(samp_dict[self.aim_str]).to(self.device)
                abs_diff.append(torch.abs(real_v - pred_v))
                abs_relate_err_value = Metric.pred_err(real_v, pred_v, epsilon)
                abs_relate_err_list.append(abs_relate_err_value)

                # deal with Nan and inf value:
                if np.isnan(abs_relate_err_value.detach()).any() or np.isinf(abs_relate_err_value.detach()).any(): print("feat_vec", samp_dict['feat_vec']); print("pred_time", pred_v); print("total_time", real_v)
                
                all_real_value = real_v if all_real_value is None else torch.cat([real_v, all_real_value])
                all_pred_value = pred_v if all_pred_value is None else torch.cat([pred_v, all_pred_value])
                
                curr_rq = Metric.r_q(real_v, pred_v, epsilon)
                curr_mean_mae = Metric.mean_mae(real_v, pred_v, epsilon)
                total_mean_mae += curr_mean_mae * len(real_v)

                if epoch % 10 == 0:
                    t_l_ = torch.mean(torch.abs(real_v - pred_v)).item()
                    p_e_ = torch.mean(abs_relate_err_value).item()
                    r_q_ = curr_rq
                    w_m_ = curr_mean_mae
                    a_e_ = Metric.accumulate_err(real_v, pred_v, epsilon)
                    print(f"####### eval by temp: idx {idx}, mean_abs_diff {t_l_}, mean_abs_relate_err {p_e_}, " +
                          f"rq {r_q_}, weighted mae {w_m_}, accumulate_err {a_e_} ")
                   
            # end if (test)
        # end loop

        if not self.test:# 计算这个batch的总 loss，for training
            self.curr_batch_losses_dict = {operator: torch.mean(torch.cat(batch_total_losses_dict[operator])).item() for operator in self.opt_obj}
            self.batch_total_loss_value = torch.mean(batch_total_loss_value / self.batch_size)

        else: # for test
            self.mean_abs_diff = torch.mean(torch.cat(abs_diff))
            self.mean_abs_relate_err = torch.mean(torch.cat(abs_relate_err_list))

            self.rq = Metric.r_q(all_real_value, all_pred_value, epsilon)
            self.accumulate_err = Metric.accumulate_err(all_real_value, all_pred_value, epsilon)
            self.weighted_mae = total_mean_mae / data_size
            self.test_loss_value = torch.mean(batch_total_loss_value / self.batch_size).item()
            if epoch % 10 == 0:
                print(f"test loss:{self.test_loss_value}")
                print(f"test batch Pred Err: {self.mean_abs_relate_err}, R(q): {self.rq}, Accumulated Error: " + 
                      f"{self.accumulate_err}, Weighted MAE: {self.weighted_mae}")


    def backward(self):
        self.last_total_loss = self.batch_total_loss_value.item()
        if self.best > self.batch_total_loss_value.item():
            self.best = self.batch_total_loss_value.item()
            self.save_units('best')
        self.batch_total_loss_value.backward()
        self.batch_total_loss_value = None

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False
        self._forward(epoch) # predict and compute the loss 

        for operator in self.optimizers: # clear prev grad first
            self.optimizers[operator].zero_grad()

        self.backward() # BP

        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()
        
    def validation(self, epoch):
        # validation 
        self.input = self.test_dataset
        self.test = True
        self._forward(epoch)
        self.last_mean_abs_diff = self.mean_abs_diff.item()
        self.last_mean_abs_relate_er = self.mean_abs_relate_err.item()
        self.last_rq = self.rq
        self.mean_abs_diff, self.mean_abs_relate_err = None, None
        self.rq = 0

    def evaluate(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self._forward(0)
        self.last_mean_abs_diff = self.mean_abs_diff.item()
        self.last_mean_abs_relate_er = self.mean_abs_relate_err.item()
        self.last_rq = self.rq
        self.mean_abs_diff, self.mean_abs_relate_err = None, None
        self.rq = 0
    

    def get_current_losses(self):
        return self.curr_batch_losses_dict

    


class QPPNet_t(QPPNet):
    def __init__(self, opt):
        super(QPPNet_t, self).__init__(opt)
        self.aim_str = 'total_time'
        self.type = 'latency'
        self.opt_obj = list(self.dim_dict.keys())

    
    def _forward_oneQ_batch(self, samp_batch): # 进行预测，并记录损失
        input_vec = torch.from_numpy(samp_batch['feat_vec']).to(self.device)
        # 递归调用，得到children plan的output
        # samp_batch['feat_vec'] + child_output_vec 即为本节点的 input
        
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _t_ = self._forward_oneQ_batch(child_plan_dict)
            input_vec = torch.cat((input_vec, child_output_vec),axis=1)
                   
        output_vec = self.units[samp_batch['node_type']](input_vec)
        # output的第一个维度即为预测的latency
        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device))

        # 计算总共延时
        pred_time = torch.sum(pred_time, 1)
        loss = (pred_time -
                torch.from_numpy(samp_batch['total_time']).to(self.device)) ** 2
        self.acc_loss[samp_batch['node_type']].append(loss)


        # added to deal with NaN
        try:
            assert(not (torch.isnan(output_vec).any()))
        except:
            print("feat_vec", samp_batch['feat_vec'], "input_vec", input_vec)
            if torch.cuda.is_available():
                print(samp_batch['node_type'], "output_vec: ", output_vec, self.units[samp_batch['node_type']].module.cpu().state_dict())
            else:
                print(samp_batch['node_type'], "output_vec: ", output_vec, self.units[samp_batch['node_type']].cpu().state_dict())
            exit(-1)
        return output_vec, pred_time
    
    def save_units(self, epoch):
        save_dir = os.path.join(self.save_dir, self.type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, unit in self.units.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(save_dir, save_filename)

            if torch.cuda.is_available():
                torch.save(unit.module.cpu().state_dict(), save_path)
                unit.to(self.device)
            else:
                torch.save(unit.cpu().state_dict(), save_path)

    def load(self, epoch):
        for name in self.units:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, self.type)
            save_path = os.path.join(save_path, save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))

            self.units[name].load_state_dict(torch.load(save_path))


class QPPNet_m(QPPNet):
    def __init__(self, opt):
        super(QPPNet_m, self).__init__(opt)
        self.aim_str = 'memory_used'
        self.type = 'memory'
        self.opt_obj = ['Memory']
    
    def _forward_oneQ_batch(self, samp_batch): # 进行预测，并记录损失
        feat_vec = samp_batch['feat_vec']
        
        input_vec = torch.from_numpy(feat_vec).to(self.device)
        # print(samp_batch['node_type'], input_vec)
        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _m_ = self._forward_oneQ_batch(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec),axis=1)
                # first dim is subbatch_size
            else:
                subplans_time.append(torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long)))

        expected_len = self.dim_dict[samp_batch['node_type']]
        if expected_len > input_vec.size()[1]:
            add_on = torch.zeros(input_vec.size()[0], expected_len - input_vec.size()[1])
            print(samp_batch['real_node_type'], input_vec.shape, expected_len)
            input_vec = torch.cat((input_vec, add_on), axis=1)

        # print(samp_batch['node_type'], input_vec)
        output_vec = self.units[samp_batch['node_type']](input_vec)
        pred_mem = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long))[:, 0]
        
        if "memory_used" in samp_batch.keys(): ## 说明此节点是root
            mem_loss = (pred_mem - 
                torch.from_numpy(samp_batch['memory_used']).to(self.device)) ** 2
            self.acc_loss["Memory"].append(mem_loss)

        
        # added to deal with NaN
        try:
            assert(not (torch.isnan(output_vec).any()))
        except:
            print("feat_vec", feat_vec, "input_vec", input_vec)
            if torch.cuda.is_available():
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].module.cpu().state_dict())
            else:
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].cpu().state_dict())
            exit(-1)

        return output_vec, pred_mem

    
    def save_units(self, epoch):
        save_dir = os.path.join(self.save_dir, self.type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, unit in self.units.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(save_dir, save_filename)

            if torch.cuda.is_available():
                torch.save(unit.module.cpu().state_dict(), save_path)
                unit.to(self.device)
            else:
                torch.save(unit.cpu().state_dict(), save_path)

    def load(self, epoch):
        for name in self.units:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, self.type)
            save_path = os.path.join(save_path, save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))

            self.units[name].load_state_dict(torch.load(save_path))