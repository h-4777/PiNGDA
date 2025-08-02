"""
This module includes code adapted from the ADGCL project:
https://github.com/susheels/adgcl

License: MIT License
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset

class TUEvaluator:
    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = 'accuracy'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'accuracy':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def _eval_accuracy(self, y_true, y_pred):
        '''
            compute Accuracy score averaged across tasks
        '''
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            acc = accuracy_score(y_true[is_labeled], y_pred[is_labeled])
            acc_list.append(acc)

        return {'accuracy': sum(acc_list) / len(acc_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_accuracy(y_true, y_pred)
    
def get_tudataset(path, name):
    assert name in ['MUTAG', 'NCI1', 'PROTEINS', 'REDDIT-BINARY', 'DD', 'COLLAB']
    
    return TUDataset(root=path, name=name)