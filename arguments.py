import argparse
import os

from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    
    parser.add_argument('--no_noise', action='store_true',
                        help='Whether use pi-noise generator.')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disables CUDA training.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size for train')
    parser.add_argument('--seed', type=float, default=None,
                        help='Seed of random number')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_compute', action="store_true")
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--repeat', type=int, default=10,
                        help='repeating times')
    parser.add_argument('--output_path', type=str, default='/output/PiNGDA/trained_models/',
                        help='The path of outputs')
    parser.add_argument('--loss_log', type=int, default=1,
                    help='Frequency to log loss.')
    parser.add_argument('--eval', type=int, default=10,
                    help='Frequency to eval model.')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset_path', type=str, default='../datasets/')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--proj_hidden_dim', type=int, default=256)
    parser.add_argument('--drop_feature', action='store_true', help="drop feature")
    parser.add_argument('--drop_edge', action='store_true', help="drop edge")
    parser.add_argument('--learnable_edge_drop', action='store_true')
    parser.add_argument('--learnable_feat_drop', action='store_true')
    parser.add_argument('--feature_drop_rate_1', type=float, default=0.15)
    parser.add_argument('--feature_drop_rate_2', type=float, default=0.15)
    parser.add_argument('--edge_drop_rate_1', type=float, default=0.2)
    parser.add_argument('--edge_drop_rate_2', type=float, default=0.2)
    parser.add_argument('--learnable_edge_lr', type=float, default=0.0001)
    parser.add_argument('--learnable_feature_lr', type=float, default=0.001)
    parser.add_argument('--learnable_edge_wd', type=float, default=0.0001)
    parser.add_argument('--learnable_feature_wd', type=float, default=0.0001)
    parser.add_argument('--eval_epoch', type=int, default=3000)
    parser.add_argument('--eval_lr', type=float, default=0.01)
    parser.add_argument('--eval_weight_decay', type=float, default=0.0)
    parser.add_argument('--temperature', type=int, default=0.3) 
    args = parser.parse_args()
    args.output_path = args.output_path + args.dataset + '/in-progress_'+datetime.now().strftime('%m%d_%H:%M:%S')

    for i in range(args.repeat):
        path = args.output_path + '/' + str(i+1)
        if not os.path.exists(path):
            os.makedirs(path)
    f = open(args.output_path+'/args.txt', "a")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}", file=f)
    f.close()
    f = open(args.output_path+'/result.txt', "a")
    f.close()
    return args
args = get_args()
print(args.output_path)