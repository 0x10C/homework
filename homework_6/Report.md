# <center> Homework 6</center>
## 1.实现细节
&emsp;&emsp;主要参考了原文的模型，原文是基于0.4.0版本的pytorch，并使用了cuda加速。提交的code基于1.0.0版本pytorch对原代码进行了修改，未使用cuda加速。
样本切分比列：训练集75%，验证集12.5%，测试集12.5%。
Batch size取1024，优化器选择AdaGrad ，dropout比列取0.2，learning rate取0.1，weight decay取5e-4，epoch取500 ，验证集check point取10,，random seed取42。两层GCN的hidden units都取128，两层GAT的hidden units 取16，多头个数取8，attention drop 为0。

## 2.结果对比分析
**在weibo数据集上的结果对比:**

来源|方法|AUC|Prec|Rec|F1
-|:-:|:-:|:-:|:-:|-
原文|GCN|76.85|42.44|71.30|53.21|
实际实验|GCN|71.37|37.67|69.21|48.79|
原文|GAT|82.72|48.53|76.00|59.27|
实际实验|GAT|50.37|25.00|99.93|40.00|


实验的机器参数为：32核128G,未使用cuda加速。GCN500个epoch大概花费了90分钟，GAT大概花费了330分钟。
## 3.改进

&emsp;&emsp;**方案一：** 原文中提到了*DeepInf*在*random walk*的时候没有考虑*side information*，可以考虑在*Deepwalk*任务中显式加入一些约束条件，这些约束调节可以是根据实际业务数据的硬规则，但是带来的后果是*sample*的速度会下降并且计算复杂度也会成倍提升。

&emsp;&emsp;**方案二：** *Ensemble*的方法，选择五个GCN,六个GAT模型，分别选择不同的random seed，在相同的训练集和参数下进行训练，二分类预测时，进行投票，



## 4.Reference
*DeepInf:Social Influence Prediction with Deep Learning*,2018, *Jiezhong Qiu, Jie Tang*

[DeepInf by xptree](github.com/xptree/DeepInf)
