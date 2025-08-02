import os
import numpy as np
import torch
from sklearn.svm import LinearSVC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.datasets import TUDataset
from pi_noise_generator import EdgeNoiseGenerator, FeatNoiseGenerator
from tu_dataset import TUEvaluator
from tu_eval import EmbeddingEvaluation
from tu_models import Model, TUEncoder
from tu_utils import initialize_edge_weight, set_tu_dataset_y_shape, drop_edges, drop_feature
from tu_arguments import get_args

def run(edge_noise_generator, feat_noise_generator, tu_encoder, dataset, args):

    device = args.device
    evaluator = TUEvaluator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(tu_encoder, args.emb_dim).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    edge_noise_optimizer = torch.optim.Adam(edge_noise_generator.parameters(),lr=0.001, weight_decay=0.0001)
    feat_noise_optimizer = torch.optim.Adam(feat_noise_generator.parameters(),lr=0.001, weight_decay=0.0001)
    ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator,
                             device, param_search=True)

    model_losses = []
    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epoch + 1):
        model_loss_all = 0

        for batch in dataloader:
            batch = batch.to(device)
            model.train()
            edge_noise_generator.train()
            model.zero_grad()

            if args.learnable_edge_drop:
                edge_noise_generator.zero_grad()
            if args.learnable_feat_drop:
                feat_noise_optimizer.zero_grad()
            if args.drop_feature:
                x_1 = drop_feature(batch.x, args.feature_drop_rate_1)
                x_2 = drop_feature(batch.x, args.feature_drop_rate_2)
            elif args.learnable_feat_drop:
                mu1, variance1 = feat_noise_generator(batch.x)
                mu2, variance2 = feat_noise_generator(batch.x)
                noise1 = feat_noise_generator.sampling(mu1, variance1)
                noise2 = feat_noise_generator.sampling(mu2, variance2)
                x_1 = batch.x + noise1
                x_2 = batch.x + noise2
            else:
                x_1 = batch.x
                x_2 = batch.x
    
            if args.drop_edge:
                e1 = drop_edges(batch.edge_index, args.edge_drop_rate_1)  
                e2 = drop_edges(batch.edge_index, args.edge_drop_rate_2)
            else:
                e1 = batch.edge_index
                e2 = batch.edge_index
            x, _ = model(batch.batch, x_1, e1, None)
            if args.learnable_edge_drop:
                probs = edge_noise_generator(batch.x, batch.edge_index)
                noise_adj = edge_noise_generator.apply_gumbel_softmax(probs, args.device)
                x_aug, _ = model(batch.batch, x_2, batch.edge_index, noise_adj)
            else:
                x_aug, _ = model(batch.batch, x_2, e2)


            model_loss = model.cal_loss(x, x_aug)
            model_loss_all = torch.sum(model_loss)

            model_loss_all.backward()
            model_optimizer.step()
            if args.learnable_edge_drop:
                edge_noise_optimizer.step()
            if args.learnable_feat_drop:
                feat_noise_optimizer.step()
        fin_model_loss = model_loss_all 
        print('Epoch {}, Model Loss {}'.format(epoch, fin_model_loss))
        model_losses.append(fin_model_loss)
        if epoch % args.eval == 0:
            model.eval()

            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
            print("Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score))


            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)


    print('FinishedTraining!')
    print('BestEpoch: {}'.format(best_val_epoch))
    print('BestTrainScore: {}'.format(best_train))
    print('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    print('FinalTestScore: {}'.format(test_curve[best_val_epoch]))


    return valid_curve[best_val_epoch], test_curve[best_val_epoch]



if __name__ == '__main__':

    args = get_args()
    my_transforms = Compose([initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset(args.dataset_path, args.dataset, transform=my_transforms)
    data = dataset[0]

    input_dim = data.x.size(1) 
    edgenoisegenerator = EdgeNoiseGenerator(input_dim).to(args.device)
    featnoisegenerator = FeatNoiseGenerator(input_dim, args.hidden_dim).to(args.device)    
    tu_encoder = TUEncoder(num_dataset_features=input_dim, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type)
    acc = []
    for i in range(args.repeat):
        ValidationScore, ac = run(edgenoisegenerator, featnoisegenerator, tu_encoder, dataset, args)
        acc.append(ac*100)
    acc = np.array(acc)
    print("max:{:.2f}, min:{:.2f}, mean:{:.2f}, std:{:.2f}".format(acc.max(), acc.min(), acc.mean(), acc.std()))   
    f = open(args.output_path+'/result.txt',"a")
    print(acc, file=f)
    print("mean:{:.2f}, std:{:.2f}".format(acc.mean(), acc.std()), file=f)
    dir = args.output_path.replace('in-progress', 'completed')
    os.rename(args.output_path, dir)
    print("result saved to" + dir)