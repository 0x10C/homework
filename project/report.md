# Project Report

## 一、MinerRL比赛及数据集总结
&nbsp;&nbsp;&nbsp;&nbsp;AlphaGo的出现兴起带动了强化学习的深入研究，由于训练数据获成本高，训练数据结构复杂等原因，许多对于深度强化学习的研究开始出现瓶颈。为了解决这个问题，各种游戏的模拟数据集相继提出，包括starcraft2，kitti，atari，minecraft等。由于Malmo的进展，各种研究minecraft为背景的数据集相继提出，但是在MinerRL数据集提出前，各种关于minecraft的数据集都还是存在比较多的缺陷，比如动作都限制为二维平面的动作，相关位置信息都是离散的，并且设计的数据集中的地图信息都比较简单，和真实minecraft的游戏地图存在一定的差距。

&nbsp;&nbsp;&nbsp;&nbsp;MinerRL数据集提出并实现了一整套的完整获取实际游戏数据的数据流程，并且是以插件的形式直接嵌入在游戏中，在游戏过程中可以获取游戏视频并且可以获得每帧的元数据信息，还可以获得游戏玩家的属性信息（生命值，等级，成就等）。而且可以利用相关的api去标注相关数据。总的来说，MinerRL任务是获取钻石，并且包括相关的六个子任务。

## 二、实验结果分析及相关方法总结
&nbsp;&nbsp;&nbsp;&nbsp;本次作业的物理机配置是32核128GB内存，1080TiGPU显存11GB

&nbsp;&nbsp;&nbsp;&nbsp;A.Proximal Policy Optimization Algorithm

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这是一种新的policy gradient的方法，
主要包括两个在下面两个步骤之间相互迭代：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.环境交互采样

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.利用梯度上升的方法对代替目标函数进行优化。

&nbsp;&nbsp;&nbsp;&nbsp;B.Imitation Learning
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

来源|方法|数据集|平均分数|方差|训练时间
-|:-:|:-:|:-:|:-:|-
baseline|ppo|MinerRLTreechop|56.31|8.31|NaN
baseline|ppo|MinerRLNavigate|8.00|27.13|NaN
实际实验|ppo|MinerRLTreechop|32.4|12.8|20h
实际实验|ppo|MinerRLNavigate|4|18.2|18h
实际实验|imitation learning |MinerRLObtainDiamond|46|21.2|80h(未跑完)

## 三、参考文献
1.<b>Proximal Policy Optimization Algorithms</b> ,John Schulman et al,2017

2.<b>Scaling Imaitation Learning In Minecraft</b> ,Artemij Amiranashvili,2020

3.<b>MinerRL:A Large-Scale Dateset of Minecraft Demonstrations</b>, William H.Goss et al,2019

4.<b>NeurIPS 2019 Competition:The MinerRL Competition on Sample Efficient reinforcement Learning using Human Priors</b>, William H.Goss et al,2019

