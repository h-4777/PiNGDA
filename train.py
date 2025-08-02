import os
import torch
import numpy as np
import utils
from load_data import get_dataset
from models import Model
from utils import drop_edges, drop_feature
from pi_noise_generator import EdgeNoiseGenerator, FeatNoiseGenerator
from linear_evaluation import linear_evaluation
from arguments import get_args

      
def train(edge_noise_generator, feat_noise_generator, data, args):
    model = Model(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_noise_optimizer = torch.optim.Adam(edge_noise_generator.parameters(),lr=args.learnable_edge_lr, weight_decay=args.learnable_edge_wd)
    feat_noise_optimizer = torch.optim.Adam(feat_noise_generator.parameters(),lr=args.learnable_feature_lr, weight_decay=args.learnable_feature_wd)
    best_ac = 0.
    
    for epoch in range(args.epoch):
        model.train()
        edge_noise_generator.train()
        optimizer.zero_grad()
        if args.learnable_edge_drop:
            edge_noise_generator.zero_grad()
        if args.learnable_feat_drop:
            feat_noise_optimizer.zero_grad()
        if args.drop_feature:
            x_1 = drop_feature(data.x, args.feature_drop_rate_1)
            x_2 = drop_feature(data.x, args.feature_drop_rate_2)
        elif args.learnable_feat_drop:
            mu1, variance1 = feat_noise_generator(data.x)
            mu2, variance2 = feat_noise_generator(data.x)
            noise1 = feat_noise_generator.sampling(mu1, variance1)
            noise2 = feat_noise_generator.sampling(mu2, variance2)
            x_1 = data.x + noise1
            x_2 = data.x + noise2
        else:
            x_1 = data.x
            x_2 = data.x
 
        if args.drop_edge:
            e1 = drop_edges(data.edge_index, args.edge_drop_rate_1)  
            e2 = drop_edges(data.edge_index, args.edge_drop_rate_2)
        else:
            e1 = data.edge_index
            e2 = data.edge_index
        
        _, z1 = model(x_1, e1)
        if args.learnable_edge_drop:
            probs = edge_noise_generator(data.x, data.edge_index)
            noise_adj = edge_noise_generator.apply_gumbel_softmax(probs, args.device)
            _, z2 = model(x_2, data.edge_index, noise_adj)   
        else:
            _, z2 = model(x_2, e2)
        loss = model.loss(z1, z2, args.batch_compute)
        
        loss.backward()
        optimizer.step()
        if args.learnable_edge_drop:
            edge_noise_optimizer.step()
        if args.learnable_feat_drop:
            feat_noise_optimizer.step()
        if (epoch+1) % 1 == 0:
            print('{}-th epoch training, '.format(epoch), end=' ')
            print('loss={}'.format(loss.item()))
            path = args.output_path + "/loss.txt"
            f = open(path, "a")
            print("Train epoch {}, loss: {:.4f}".format(epoch + 1, loss.item()), file=f)
            f.close()
            model.train()
            if args.learnable_edge_drop:
                edge_noise_generator.train()
            if args.learnable_feat_drop:
                feat_noise_generator.train()
        if (epoch+1) % args.eval == 0:
            print('{}-th epoch training, '.format(epoch+1), end=' ')
            model.eval()
            z = model(data.x, data.edge_index)[0].detach()
            y = data.y
            output, ac= linear_evaluation(z, y, args, epoch + 1)
            if ac > best_ac:
                best_ac = ac  
            path = args.output_path + "/eval.txt"
            f = open(path, "a")
            print(output, file=f)
            f.close()          
    print("best ac: {:.4f}".format(best_ac))
    return best_ac




if __name__ == "__main__":
    args = get_args()
    if args.seed is not None:
        utils.set_seed(args.seed)
    data = get_dataset(args.dataset_path, args.dataset).to(args.device)
    input_dim = data.num_features
    output_dim = data.num_features
    edgenoisegenerator = EdgeNoiseGenerator(input_dim).to(args.device)
    featnoisegenerator = FeatNoiseGenerator(input_dim, args.hidden_dim).to(args.device)
    acc = []
    for i in range(args.repeat):
        ac = train(edgenoisegenerator, featnoisegenerator, data, args)
        acc.append(ac)
    acc = np.array(acc)
    print("max:{:.2f}, min:{:.2f}, mean:{:.2f}, std:{:.2f}".format(acc.max(), acc.min(), acc.mean(), acc.std()))   
    f = open(args.output_path+'/result.txt',"a")
    print(acc, file=f)
    print("mean:{:.2f}, std:{:.2f}".format(acc.mean(), acc.std()), file=f)
    dir = args.output_path.replace('in-progress', 'completed')
    os.rename(args.output_path, dir)
    print("result saved to" + dir)