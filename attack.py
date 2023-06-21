# from config import *
# from utils import *
# from models import GCN, PTDNetGCN
# from metrics import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# # Settings
# dataset_name='cora'
# args.dataset=dataset_name
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
# all_labels = y_train + y_test+y_val
# single_label = np.argmax(all_labels,axis=-1)
# nodesize = features.shape[0]

# # Some preprocessing
# features_tmp=features.copy()
# features = preprocess_features(features).A
# support = preprocess_adj(adj)

# optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

# tuple_adj = sparse_to_tuple(adj.tocoo())
# features_tensor = tf.convert_to_tensor(features,dtype=dtype)
# adj_tensor = tf.SparseTensor(*tuple_adj)
# y_train_tensor = tf.convert_to_tensor(y_train,dtype=dtype)
# train_mask_tensor = tf.convert_to_tensor(train_mask)
# y_test_tensor = tf.convert_to_tensor(y_test,dtype=dtype)
# test_mask_tensor = tf.convert_to_tensor(test_mask)
# y_val_tensor = tf.convert_to_tensor(y_val,dtype=dtype)
# val_mask_tensor = tf.convert_to_tensor(val_mask)

# best_test_acc = 0
# best_val_acc_trail = 0
# best_val_loss = 10000

# from deeprobust.graph.data import Dataset
# from deeprobust.graph.defense import GCN
# from deeprobust.graph.global_attack import Metattack
# # Setup Surrogate model
# idx_train=np.array(np.where(train_mask==1)).tolist()[0]
# idx_val=np.array(np.where(val_mask==1)).tolist()[0]
# idx_test=np.array(np.where(test_mask==1)).tolist()[0]
# idx_unlabeled = np.union1d(idx_val,idx_test)
# surrogate = GCN(nfeat=features.shape[1], nclass=single_label.max().item()+1,weight_decay=0.0,
#                                 nhid=256, dropout=0, with_relu=True, with_bias=False, device='cpu').to('cpu')
# surrogate.fit(features, adj, single_label, idx_train, idx_val, patience=100)
# # Setup Attack Model
# model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
#         attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
# # Attack
# model.attack(features, adj, single_label, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
# modified_adj = model.modified_adj
# # print(adj)
# # print("shiy")
# # print(modified_adj)

# modified_adj=sp.csr_array(modified_adj.int())

# from config import args
# import tensorflow as tf
# import time
# from utils import *
# from models import GCN_dropedge
# from metrics import *

# # Settings
# optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
# args.dataset=dataset_name

# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
# del adj
# adj=modified_adj

# tuple_adj = sparse_to_tuple(adj.tocoo())
# adj_tensor = tf.SparseTensor(*tuple_adj)

# features = preprocess_features(features)

# model = GCN_dropedge(input_dim=features.shape[1], output_dim=y_train.shape[1], adj=adj_tensor)


# features_tensor = tf.convert_to_tensor(features,dtype=tf.float32)
# y_train_tensor = tf.convert_to_tensor(y_train,dtype=tf.float32)
# train_mask_tensor = tf.convert_to_tensor(train_mask)
# y_test_tensor = tf.convert_to_tensor(y_test,dtype=tf.float32)
# test_mask_tensor = tf.convert_to_tensor(test_mask)
# y_val_tensor = tf.convert_to_tensor(y_val,dtype=tf.float32)
# val_mask_tensor = tf.convert_to_tensor(val_mask)

# best_test_acc = 0
# best_val_acc = 0
# best_val_loss = 10000


# curr_step = 0
# for epoch in range(args.epochs):

#     with tf.GradientTape() as tape:
#         output = model.call((features_tensor),training=True)
#         cross_loss = masked_softmax_cross_entropy(output, y_train_tensor,train_mask_tensor)
#         lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
#         loss = cross_loss #+ args.weight_decay*lossL2
#         grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     output = model.call((features_tensor), training=False)
#     train_acc = masked_accuracy(output, y_train_tensor,train_mask_tensor)
#     val_acc  = masked_accuracy(output, y_val_tensor,val_mask_tensor)
#     val_loss = masked_softmax_cross_entropy(output, y_val_tensor, val_mask_tensor)
#     test_acc  = masked_accuracy(output, y_test_tensor,test_mask_tensor)

#     if val_acc > best_val_acc:
#         curr_step = 0
#         best_test_acc = test_acc
#         best_val_acc = val_acc
#         best_val_loss= val_loss
#         # Print results

#     else:
#         curr_step +=1
#     if curr_step > args.early_stop:
#         print("Early stopping...")
#         break

#     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cross_loss),"val_loss=", "{:.5f}".format(val_loss),
#       "train_acc=", "{:.5f}".format(val_acc), "val_acc=", "{:.5f}".format(val_acc),
#       "test_acc=", "{:.5f}".format(best_test_acc))


import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.01,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=256,
        dropout=0.0, with_relu=True, with_bias=False, weight_decay=0, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=256,
        dropout=0.0, with_relu=True, with_bias=False, weight_decay=0, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    modified_adj = model.modified_adj
    # modified_features = model.modified_features
    test(modified_adj)

    gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=256,
        dropout=0.0, with_relu=True, with_bias=False, weight_decay=0, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, modified_adj, labels, idx_train)
    gcn.test(idx_unlabeled)
    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()