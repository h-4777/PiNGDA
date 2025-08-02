import argparse
import os

from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='DD',
                        help='Dataset')
    parser.add_argument('--dataset_path', type=str, default='../datasets/')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model Learning rate.')    
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='Model Weight Decay.')
    parser.add_argument('--noise_edge_lr', type=float, default=0.01)
    parser.add_argument('--noise_feat_lr', type=float, default=0.001)
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--batch_compute', action="store_true")
    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--repeat', type=int, default=5,
                        help='repeating times')
    parser.add_argument('--output_path', type=str, default='/output/tu_PiNGDA/trained_models/',
                        help='The path of outputs')
    parser.add_argument('--loss_log', type=int, default=1,
                    help='Frequency to log loss.')
    parser.add_argument('--eval', type=int, default=5,
                    help='Frequency to eval model.')
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