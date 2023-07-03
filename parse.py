import argparse

parser = argparse.ArgumentParser(description='QPPNet Arg Parser')
# Environment arguments
# required
parser.add_argument('--data_dir', type=str, default='./data/pgdata',
                    help='Dir containing train data')

parser.add_argument('--type', type=str, default='latency',
                    help='latency prediction of memory prediction')

parser.add_argument('--dataset', type=str, default='PSQLTPCH',
                    help='Select dataset [PSQLTPCH | TerrierTPCH | OLTP]')

parser.add_argument('--test_time', action='store_true',
                    help='if in testing mode')

parser.add_argument('-dir', '--save_dir', type=str, default='./saved_model',
                    help='Dir to save model weights (default: ./saved_model)')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')

parser.add_argument('--scheduler', action='store_true', default='True') # default='False'
parser.add_argument('--step_size', type=int, default=30,
                    help='step_size for StepLR scheduler (default: 1000)')

parser.add_argument('--gamma', type=float, default=0.95,
                    help='gamma in Adam (default: 0.95)')

parser.add_argument('--SGD', action='store_true', default='True', # default=false
                    help='Use SGD as optimizer with momentum 0.9')



parser.add_argument('--batch_size', type=int, default=128, 
                    help='Batch size used in training (default: 32)')

parser.add_argument('-s', '--start_epoch', type=int, default=0,
                    help='Epoch to start training with (default: 0)')

parser.add_argument('-t', '--end_epoch', type=int, default=250000,
                    help='Epoch to end training (default: 200)')

parser.add_argument('-epoch_freq', '--save_latest_epoch_freq', type=int, default=100) 

parser.add_argument('-logf', '--logfile', type=str, default='train_loss.txt')

parser.add_argument('--mean_range_dict', type=str)

def save_opt(opt, logf):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    logf.write(message)
    logf.write('\n')