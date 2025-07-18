import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
# import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CausalPD: Joint Causal Discovery and Intervention for Large-Scale Pavement Distress Distribution Data')

    # random seed
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='CausalPD',
                        help='model name, options: [CausalPD]')

    # data loader
    parser.add_argument('--data', type=str, default='train', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/Shanghai/', help='root path of the data file, options: [GridSZ, SegmentSZ, Shanghai]')
    parser.add_argument('--data_path', type=str, default='pavement_distress.npy', help='data file')
    parser.add_argument('--ext_path', type=str, default='ext.csv', help='data file')
    parser.add_argument('--meta_dim', type=int, default=0, help='dimension of static spatial features, set to 0 to disable static feature loading')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=28, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=14, help='start token length')
    parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length')

    # CausalPD
    parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # intervention training
    parser.add_argument('--use_intervention', type=int, default=1, help='whether to use intervention training')
    parser.add_argument('--intervention_epochs', type=int, default=50, help='number of epochs for intervention training')
    parser.add_argument('--attention_threshold', type=float, default=0.5, help='threshold for separating causal and non-causal patches')
    parser.add_argument('--entropy_weight', type=float, default=0.05, help='weight for entropy regularization')
    parser.add_argument('--consistency_weight', type=float, default=0.1, help='weight for consistency loss')
    parser.add_argument('--K', type=int, default=4, help='number of intervention samples')
    parser.add_argument('--intervention_prob', type=float, default=0.7, help='probability of applying intervention in each batch')
    parser.add_argument('--intervention_lr_scale', type=float, default=0.1, help='learning rate scale for intervention training')
    parser.add_argument('--intervention_lr_decay', type=float, default=0.95, help='learning rate decay factor for intervention training')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    #wandb
    # wandb.init(project="prediction", config=args)
    # config = wandb.config

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = (
                f"{args.model_id}_{args.model}_{args.data}_"
                f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
                f"lr{args.learning_rate}_bs{args.batch_size}_intv{args.use_intervention}_"
                f"{args.des}_{ii}"
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = (
            f"{args.model_id}_{args.model}_{args.data}_"
            f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
            f"lr{args.learning_rate}_bs{args.batch_size}_intv{args.use_intervention}_"
            f"{args.des}_{ii}"
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        