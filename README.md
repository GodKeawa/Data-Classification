# Data-Classification
本项目是**FDU特色**思政-践行平台课题——医保、医疗、医药协同发展和治理研究的AI训练子项目  
项目仍在完善中  
**Tips: 深度学习的项目不是很好复现，且本项目的库版本都很新，不好debug**

# 项目环境

- 系统：win11专业版
- 显卡：RTX4060 Desktop
  - 显卡驱动版本：561.09
- Cuda：https://developer.nvidia.com/cuda-12-4-1-download-archive
- Cudnn:https://developer.nvidia.com/cudnn-9-1-1-download-archive
- 软件环境：
  - PyCharm professional
  - Anaconda (Latest)
  - python: Anaconda虚拟环境选3.11
  - Pytorch和附属 (去官网找cuda对应的版本然后下载)
    - 我的：conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
  - 其他包没有冲突，任意版本均可，放在requirements.txt里，PyCharm可以一键安装，我的是
    - Transformers 4.44.1
    - scikit-learn 1.5.2

# 项目简介
非常普通和简单的一个文本分类库，参考了一些老的框架代码，框架的细节不必太过考究  
可以参考代码注释进行理解，建议优先看一下Ref里的翻译后的论文，大致结构相同

使用与参考论文相同的预训练Bert模型进行再训练，使用scikit-learn的学习框架搭建迭代训练系统  
之后只要调一下参，多测试一下就行

项目使用了新闻标准数据集做框架有效性测试  
最终效果相当不错，在learning_rate为5e-6时只要5次迭代即可有95%的正确率

本项目由于现在只有我一人标注数据，所以数据集很小，相对来说不是那么好训练  
本人的解决方案是调低learning_rate并提高迭代次数  
最后在4e-6的learning_rate下, 跑了60次迭代，模型基本收敛  
60次迭代的结果见result.txt  

另外项目目前的主题词条数还很少，这是因为部分主题缺少足够数据，无法正常训练  
在人工标注数据时预设的主题分类也还很少，后续可以细化  
但是实际上我现在的想法是舍弃分类模型，转向更为开放的文本摘要模型  
因为在数据标注的过程中可以明显发现有一些主题是经常出现的，比如  
- 中医药
- 罕见病
- 不孕不育(联动人口问题)
- 长期保健医保
- 医保支付手段
- ...  
实际上使用ai模型提取出这些主题可能更有价值，后续可以为每篇文章设置更为精准的标签  
之后便可以进行数据处理，做一些时间轴类型的研究等

# 项目运行
并不建议没有显卡的PC跑，笔记本请注意散热  
另外环境配置往往才是第一难  

先到我的onedrive或网盘下载模型文件和数据文件(onedrive需要科学上网，但是不限速)  
模型文件也可以去huggingface下载，搜索'bert-base-chinese'即可  
onedrive
models:https://1drv.ms/u/c/5feadf21a615ba59/Ec4N_7B2AkpNqXAwg1VSJY8BNONRDfgqHOM6rqdf86rrlQ?e=ItcFBT  
data:https://1drv.ms/u/c/5feadf21a615ba59/Efb_MQP9iwVDtnJMXrcHawsB129RWFmYVgIXkV5Lsa1j7g?e=xMGvnR
链接：https://pan.baidu.com/s/1oDWnF4zPo6NU_w9VvFgm3A?pwd=6p1c 
提取码：6p1c

下载完成后将文件夹放在项目根目录就行
本人已经训练好了两个模型可以拿来测试，一个是使用新闻标准数据库的，一个是本项目的  

使用时请自行调好参数，选好目录  
本项目只需要选择data_select即可，对应data和models内的文件夹  
但是如果报错了就自己检查一下

dataset.py和model.py都是辅助用的文件  
train.py进行训练和测试，请自行选择train函数和test函数  
predict.py使用训练好的模型进行分类任务  

# 待更新


