import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
print(adj.shape)
print(features.shape)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled = np.union1d(idx_val, idx_test)
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
# Attack
model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
modified_adj = model.modified_adj