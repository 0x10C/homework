# Homework_7
## 1.实现细节
&emsp;&emsp;本次作业主要是参考原文作者实现的GraphSGAN实现代码（python = 2.7，pytorch = 0.3.0）,在python = 3.6，pytorch>1.0.0的环境中复现,使用了cuda加速,实验物理机参数为32核128G内存，显卡为1080ti，显存11G。

&emsp;&emsp;分类器中有五层隐藏层，隐藏层神经元个数为（500，500，250，250，250）；生成器中有两层隐藏层，隐藏层神经元个数为（500，500），并且使用了batch normalization；输入的随机噪声使用均值为0，标准差为0.05的高斯白噪声，隐藏层使用均值为0，标准差为0.5的高斯白噪声；激活函数除了输出层均使用了elu,输出层激活函数使用tanh；所有权重初始化都使用xavie初始化；使用adam优化器。迭代算法中涉及的超参数 &lambda;<sub>0</sub> = 2，&lambda;<sub>1</sub> = 1，&lambda;<sub>2</sub>  = 0.3，&beta;<sub>1</sub> = 0.5，&beta;<sub>2</sub> = 0.999；使用了SGD momentum = 0.5;batch size 取64；epoch取20个。
## 2.性能结果分析
&emsp;&emsp;训练过程共耗时大约170s。


来源|数据集|准确率平均值|准确率标准差
-|:-:|:-:|-
论文|cora|83|1.3
复现实验|cora|81.6|2.5

从结果上看准确率平均值比论文低了1.4，标准差大了1.2，猜想可能有可能是随机游走生成graph embedding不稳定导致的，以及初始化的random seed可能与论文中的不同。

## 3.改进
&emsp;&emsp;尝试了以下方法，但是效果并没有显著提升：调整随机噪声层的均值和方差，用的是比较hard的调整，考虑尝试根据生成器和分类器对同一个数据表示的均值和方差动态调整随机噪声层的均值和方差，但是动态调整的方法缺乏理论基础，但是值得尝试一下。

## 4.参考
[1] Semi-supervised Learning on Graphs with Generative Adversarial Nets,2018,Ming Ding,Jie Tang

[2] [GraphSGAN by THUDM](https://github.com/THUDM/GraphSGAN)
