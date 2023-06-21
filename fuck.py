import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

from config import *
from utils import *
from metrics import *
###
import tensorflow as tf
import time
from models import GCN_dropedge


 # Settings




import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name='cora', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
len_tmp = labels.shape[0]
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = idx_test#np.union1d(idx_val, idx_test)


adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
print(features.shape)

ptb_rate = 0.05
perturbations = int(ptb_rate * (adj.sum()//2))

train_mask = np.zeros(len_tmp)
train_mask[idx_train] = 1

val_mask = np.zeros(len_tmp)
val_mask[idx_val] = 1

test_mask = np.zeros(len_tmp)
test_mask[idx_test] = 1

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

#tuple_adj = sparse_to_tuple(adj.tocoo())
features_tensor = tf.convert_to_tensor(features,dtype=dtype)
#adj_tensor = tf.SparseTensor(*tuple_adj)

labels_tmp = F.one_hot(labels)


y_train_tensor = tf.convert_to_tensor(labels_tmp,dtype=dtype)
train_mask_tensor = tf.convert_to_tensor(train_mask)
y_test_tensor = tf.convert_to_tensor(labels_tmp,dtype=dtype)
test_mask_tensor = tf.convert_to_tensor(test_mask)
y_val_tensor = tf.convert_to_tensor(labels_tmp,dtype=dtype)
val_mask_tensor = tf.convert_to_tensor(val_mask)

best_test_acc = 0
best_val_acc_trail = 0
best_val_loss = 10000

# Setup Surrogate model

surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,weight_decay=0.0,
                                nhid=16, dropout=0, with_relu=True, with_bias=False, device='cpu').to('cpu')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=100)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
# Attack

model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=perturbations, ll_constraint=False)
modified_adj = model.modified_adj

modified_adj=sp.csr_array(modified_adj.int())

tuple_adj = sparse_to_tuple(modified_adj.tocoo())
adj_tensor = tf.SparseTensor(*tuple_adj)


# Settings
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
model = GCN_dropedge(input_dim=features.shape[1], output_dim=labels.max().item()+1, adj=adj_tensor)


best_test_acc = 0
best_val_acc = 0
best_val_loss = 10000


curr_step = 0
for epoch in range(args.epochs):

    with tf.GradientTape() as tape:
        output = model.call((features_tensor),training=True)
        cross_loss = masked_softmax_cross_entropy(output, y_train_tensor,train_mask_tensor)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = cross_loss #+ args.weight_decay*lossL2
        grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    output = model.call((features_tensor), training=False)
    train_acc = masked_accuracy(output, y_train_tensor,train_mask_tensor)
    val_acc  = masked_accuracy(output, y_val_tensor,val_mask_tensor)
    val_loss = masked_softmax_cross_entropy(output, y_val_tensor, val_mask_tensor)
    test_acc  = masked_accuracy(output, y_test_tensor,test_mask_tensor)

    if val_acc > best_val_acc:
        curr_step = 0
        best_test_acc = test_acc
        best_val_acc = val_acc
        best_val_loss= val_loss
        # Print results

    else:
        curr_step +=1
    if curr_step > args.early_stop:
        print("Early stopping...")
        break

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cross_loss),"val_loss=", "{:.5f}".format(val_loss),
    "train_acc=", "{:.5f}".format(val_acc), "val_acc=", "{:.5f}".format(val_acc),
    "test_acc=", "{:.5f}".format(best_test_acc))