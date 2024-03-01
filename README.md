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

$$
min\mathcal{L}_{total}=\mathcal{L}_{GNN}+\textcolor[rgb]{0.7,0.5,0}{\beta Tr(X^T\Phi X)}+\textcolor[rgb]{1,0,0}{\alpha\mathcal{L}_{mask}}+\textcolor[rgb]{0,1,0}{\gamma*GNN的所有参数的l2范数}
$$

## 结果

![](.\result\result.png)
