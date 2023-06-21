from config import *
from utils import *
from models import GCN, PTDNetGCN
from metrics import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# Settings

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
all_labels = y_train + y_test+y_val
single_label = np.argmax(all_labels,axis=-1)
nodesize = features.shape[0]

# Some preprocessing
features_tmp=features.copy()
features = preprocess_features(features).A
support = preprocess_adj(adj)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

tuple_adj = sparse_to_tuple(adj.tocoo())
features_tensor = tf.convert_to_tensor(features,dtype=dtype)
adj_tensor = tf.SparseTensor(*tuple_adj)
y_train_tensor = tf.convert_to_tensor(y_train,dtype=dtype)
train_mask_tensor = tf.convert_to_tensor(train_mask)
y_test_tensor = tf.convert_to_tensor(y_test,dtype=dtype)
test_mask_tensor = tf.convert_to_tensor(test_mask)
y_val_tensor = tf.convert_to_tensor(y_val,dtype=dtype)
val_mask_tensor = tf.convert_to_tensor(val_mask)

best_test_acc = 0
best_val_acc_trail = 0
best_val_loss = 10000

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
# Setup Surrogate model
idx_train=np.array(np.where(train_mask==1)).tolist()[0]
idx_val=np.array(np.where(val_mask==1)).tolist()[0]
idx_unlabeled=np.array(np.where(test_mask==1)).tolist()[0]
surrogate = GCN(nfeat=features.shape[1], nclass=single_label.max().item()+1,
                nhid=256, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
surrogate.fit(features, adj, single_label, idx_train, idx_val, patience=30)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
# Attack
model.attack(features, adj, single_label, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
modified_adj = model.modified_adj
print(adj)
print(adj)
print("shiy")
print(modified_adj)

'''
import time
begin = time.time()

model = PTDNetGCN(input_dim=features.shape[1], output_dim=y_train.shape[1])
model.set_fea_adj(np.array(range(adj.shape[0])), features_tensor, adj_tensor)

best_epoch = 0
curr_step = 0
best_val_acc = 0

for epoch in range(args.epochs):
    temperature = max(0.05,args.init_temperature * pow(args.temperature_decay, epoch))
    with tf.GradientTape() as tape:
        preds = []
        for l in range(args.outL):
            output = model.call(temperature,training=True)
            preds.append(tf.expand_dims(output,0))
        all_preds = tf.concat(preds,axis=0)
        mean_preds = tf.reduce_mean(preds,axis=0)
        consistency_loss = tf.nn.l2_loss(mean_preds-all_preds)

        cross_loss = masked_softmax_cross_entropy(mean_preds, y_train_tensor,train_mask_tensor)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        lossl0 = model.lossl0(temperature)
        nuclear = model.my_nuclear()
        #nuclear = model.nuclear()
        loss = cross_loss + args.weight_decay*lossL2 + args.lambda1*lossl0 + args.lambda3*nuclear + args.coff_consis*consistency_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    output = model.call(None, training=False)
    edges_volumn = tf.reduce_sum(model.maskes[0])
    print('edge_vol',edges_volumn.numpy())

    train_acc = masked_accuracy(output, y_train_tensor,train_mask_tensor)
    val_acc  = masked_accuracy(output, y_val_tensor,val_mask_tensor)
    val_loss = masked_softmax_cross_entropy(output, y_val_tensor, val_mask_tensor)
    test_acc  = masked_accuracy(output, y_test_tensor,test_mask_tensor)
    if val_acc > best_val_acc:
        curr_step = 0
        best_epoch = epoch
        best_val_acc = val_acc
        best_val_loss= val_loss
        if val_acc>best_val_acc_trail:
            best_test_acc = test_acc
            best_val_acc_trail = val_acc
    else:
        curr_step +=1
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cross_loss),"val_loss=", "{:.5f}".format(val_loss),
      "train_acc=", "{:.5f}".format(train_acc), "val_acc=", "{:.5f}".format(val_acc),"best_val_acc_trail=", "{:.5f}".format(best_val_acc_trail),
      "test_acc=", "{:.5f}".format(best_test_acc))

    if curr_step > args.early_stop:
        print("Early stopping...")
        break
    end = time.time()
    print('time ',(end-begin))
'''