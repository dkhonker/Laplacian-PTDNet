# 基于PTDNet的图神经网络鲁棒性机制



## 安装需求

  * Python 3.8.6
  * tensorflow 2.3.1
  * networkx

## 运行代码

```
python3 train_PTDNet.py --dataset cora --dropout 0.0
```
注意：

cora可以改为citeseer，pubmed。

dropout 是对特征操作的，类似于dropmeassge，可以不使用，即

```
python3 train_PTDNet.py --dataset cora
```

## 核心思想

$min\mathcal{L}_{total}=\mathcal{L}_{GNN}+\beta Tr(X^T\Phi X)+\alpha\mathcal{L}_{mask}+\gamma*GNNL2$

## 结果

![](result\result.png)
