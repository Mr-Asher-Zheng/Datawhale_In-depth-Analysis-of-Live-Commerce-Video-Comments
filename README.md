# Datawhale_In-depth-Analysis-of-Live-Commerce-Video-Comments
- 比赛链接：
  [基于带货视频评论的用户洞察挑战赛](https://challenge.xfyun.cn/topic/info?type=video-comment-insight&option=ssgy&ch=dwsfsp25-1)
- 比赛手册：
  [Datawhale_大模型技术_基于带货视频评论的用户洞察挑战赛](https://www.datawhale.cn/activity/324/learn/191/4330)
- 先上分数：
  <img width="1255" height="140" alt="08fd972c4b4097d2bcfa1a639890c313" src="https://github.com/user-attachments/assets/8d2c8dc4-cd84-48c4-a17b-f52ff83e4dcf" />

## 📔日志

- 比赛时间：2025年7月6日-7月17日
- 仓库和笔记创建时间：2025年8月28号（时隔一个多月才来写完整笔记，整理屎山代码时真想给自己一巴掌😂，为什么不早点整理！！！现在只能凭着记忆写了，因为在整理过程中大多数文件的路径有更改且改完后我没有测试过（`我已经尽量改好了`），直接运行可能会有报错（`如果运行有报错可以在GitHub这里反馈一下，...如果有人看的话`），可能会有所缺漏）

## 📝 比赛思路
### 商品识别
首先是商品识别方面，我使用了调用大模型下prompt的方法
- 下prompt的技巧：
  - 1、说清楚规则
  - 2、在李宏毅老师的课程（2024、2025）中提到过，似乎模型比较能读懂模型输出的句子，那就可以先让一个大语言模型根据你的要求生成prompt，然后你再拿语言模型生成的prompt给用来预测结果的语言模型，但这部分我没实践过，自行实践或查论文求证。
  - 3、给足够多的示例（比赛数据中有标注商品类别的我都给了，一共20条）
- 模型选用技巧：
  - 1、一般来说参数量越大的模型能力越强
  - 2、根据数据选择，例如这次的数据是有关多语言的、电商的、输出数据是结构的（json），就尽量选择训练在多语言数据、电商数据、输出结构数据下的模型

### 情感分析
然后是情感分析方面，这部分花的时间最多，使用了很多新技术（对我当时拥有的知识来说）
- 数据清洗方面
    - emoji：在比赛交流会中，有的人认为emoji是杂讯，数据清洗直接去除了emoji；有的人直接把emoji当作训练数据进行训练，emoji在一些模型embedding后在向量空间中和普通文本向量的距离方向是比较接近的（`当时有个大佬说有求证过`），同类emoji之间在向量空间中是比较接近的，[引用文章：论大模型和嵌入模型对Emoji的表现](https://zhuanlan.zhihu.com/p/690813729)，简单来说就是它们被模型认为是同一个东西（比如“😀”和“笑”、“😀”和“😂”在向量空间中是比较接近的），但我不认为emoji是杂讯（我当时没了解到emoji和文本、emoji之间在向量空间中的关系），所以我的方法是将emoji转换成文本来保留这部分信息。
    - 一些简单清洗：例如将 HTML/XML 实体编码（如 \&#39; \&amp;）转换为原始字符、移除剩余HTML标签等

- 数据增强方面（因为数据量太少了，做了这部分可以提高一点点分数）
  - 翻译增强：使用讯飞API调用模型来翻译文本，因为数据大多是英文和中文（`如果我没记错的话`），我就把英语翻译成中文、其他语言翻译成英语当作新的数据。
  - EDA数据增强：查看文章了解吧：[引用文章：【工程实践】使用EDA(Easy Data Augmentation)做数据增强](https://blog.csdn.net/weixin_44750512/article/details/131675191)

- loss function
  - Focal Loss（这次项目的唯一的卡密`かみ`，能直接把loss压低，把F1提高5%）
    - 参考文章和论文：[pytorch如何使用Focal Loss](https://blog.csdn.net/weixin_45277161/article/details/132626946)、[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
    - 这里放几张我训练出来表现最好的模型（`千问+xrl+2EDA+全focal_loss+标签平滑+翻译增强`）的日志图，其实还有其他的，但是我文件命名不规范，记不得其他调什么了，所以对比不了
    - F1：
      <img width="1440" height="660" alt="image" src="https://github.com/user-attachments/assets/2c7c3dc9-b216-48b5-ba1a-b0ffacc2505d" />
      <img width="1394" height="578" alt="image" src="https://github.com/user-attachments/assets/058145e1-315c-46ed-84ad-6bba3cfca50e" />
      <img width="1394" height="620" alt="image" src="https://github.com/user-attachments/assets/4f01a738-3a36-4dfa-b8c9-7b5971fe9dea" />
    - lr：
      <img width="1436" height="729" alt="image" src="https://github.com/user-attachments/assets/c2dcd303-7ff7-462e-88e0-329be896f987" />
    - Loss：
      <img width="1434" height="674" alt="image" src="https://github.com/user-attachments/assets/bf944b07-c9d4-4c3d-aa4a-549b6764fdfa" />
    
- scheduler
  - 这里使用了带 warmup 的余弦衰减（Cosine Annealing with Warmup）学习率调度，是在[李宏毅老师机器学习2022的作业4](https://www.bilibili.com/video/BV1Wv411h7kN?spm_id_from=333.788.videopod.episodes&vd_source=108c7957d88a7eafb172b276df7068cf&p=45)中学习到的，[作业4讲解](https://www.bilibili.com/video/BV1ZWKWeKE4v/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=108c7957d88a7eafb172b276df7068cf)，具体见作业和视频讲解

- model
  - Bert：
    - 模型选型方面我最开始用的是[bert-base-multilingual-cased](https://www.modelscope.cn/models/google-bert/bert-base-multilingual-cased)(`这是标准BERT架构在多语言数据集上训练的模型`)，后来使用[xlm-roberta-base](https://www.modelscope.cn/models/AI-ModelScope/xlm-roberta-base)(`这是对BERT的训练策略和数据使用进行优化后在多语言数据集上训练的模型`)，最后使用[xlm-roberta-large](https://www.modelscope.cn/models/AI-ModelScope/xlm-roberta-large)(`参数量比xlm-roberta-base大，在此项目经过实践证明是比base好的`)
    - 网络架构：
      <img width="1796" height="413" alt="graphviz" src="https://github.com/user-attachments/assets/380b04a5-171a-4325-8529-6cf2d67d02c2" />
      
  - 大语言模型：
    - 在比赛交流会中，这部分有些人是使用大语言模型做的，由于比赛进行时我正在学习大语言模型相关知识所以没在这部分用大语言模型，而且我的笔记本电脑只能运行小的大语言模型，而且大语言模型的API调用要花钱，况且有些人用了大语言模型后的分数也没我的高（嘻嘻，当然也有一些比我高的），所以这部分我就没怎么用大语言模型

- other
  - 类别权重、标签平滑

- TODO
  - 这部分的主要技术都在这里了，改进的方面可能有这么一些：
  - 我现在的总loss是直接相加(`loss = sentiment_loss + scenario_loss + question_loss + suggestion_loss`)然后反向传播，这会导致最主要的问题就是
    - 任务不平衡：各个任务的 loss 值可能数量级差别很大，直接相加后，loss 大的任务会 dominate 梯度，模型偏向优化这个任务，其他任务学习不到好特征；
    - 解决方法：给每个任务加权（可分为固定加权和动态加权，我的代码里好像实现的是固定加权）
    - 梯度冲突：不同任务的梯度方向可能冲突（梯度指向不同方向），简单相加会让模型在某些参数上“拉扯”，可能导致：部分任务收敛慢，学习不稳定；
    - 解决方法（`chatgpt说的`）：使用 梯度投影 / 平衡 技术：PCGrad（Project Conflicting Gradients）或GradNorm（按梯度大小动态调整权重）

### 聚类洞察
最后是评论聚类，这部分的提分是最难的，在在比赛交流会中了解到这部分几乎没有什么人能得到高分（`可能是比赛评分的问题？`），而且我对聚类这种传统的机器学习不是很熟悉，所以最后也没得到太好的分数，这部分需要对评论进行正面倾向聚类、负面倾向聚类、用户场景聚类、用户疑问聚类、用户建议聚类，然后提取每个类别的前十个主题词

- 主要思路：这部分的目的是聚类和提取关键词，答案的提交是词语，所以需要对每一条评论进行分词、去停用词、去特殊符号、去url等清洗，最终每条训练数据的格式是一个个用空格分开的有意义的主题词语，例如：语言 翻译器 便宜 功能 多，然后进行数据进行embedding，然后使用K-Means或DBSCAN等技术对每个类别选择最佳聚类数（范围：5~8），根据每个类别的最佳k值，分别对每个类别的embeddings做KMeans或DBSCAN聚类，然后对每个聚类里的评论做TF-IDF统计，找出最能代表该类的top N个主题词（默认10个），把聚类主题词写回comments_data

- 数据清洗（`这里应该是提分的重点，因为数据很杂、很多无效信息，需要对数据进行彻底的清洗`）
  - 转换特殊名词：把"Xfaiyx Smart Recorder"换成"XFRS"，"Xfaiyx Smart Translator"换成"XFST"，然后进行聚类、提取，在产生答案时再将其换回原来的词语。因为这两个特殊名词是商品名词，是大多是评论都有的，这两个名词在分词的时候会被划分成3个词，比如“Xfaiyx Smart Translator”会被划分成“Xfaiyx”、“Smart”、“Translator”，这就导致提取主题词时，本来是提取一个整体一个词的“Xfaiyx Smart Translator”，变成提取成“Xfaiyx”、“Smart”、“Translator”三个词，这些词又是在大多数评论中存在的，所以你提取的前十个主题词大概率有3个跟答案是不一样的，所以得分会低。

  - 清除emoji：这部分存在争议，有的人清除了emoji后得分比较高，也有的人清除了emoji后的分会降低，可能是聚类采取的方法不同导致的差异吧，我的代码有这个功能，看情况使用吧

  - 去停用词：因为数据是多语言的，需要为每种语言准备分词器和停用词表（`没有就用默认的`）

- embedding：
  - 调用大模型API（`我的是Qwen`）生成指定评论的向量，也可以用普通模型的tokenizer进行embedding（`例如xlm-roberta-large的Tokenizer`），但是效果没大模型的好。

- 聚类：
  - 可以每种方法都试一下，每种方法各有千秋
    <img width="2100" height="1300" alt="聚类" src="https://github.com/user-attachments/assets/58898acf-cdea-4f30-b580-e2a375057d86" />

## 📂 文件说明

- `origin_data`：数据集文件夹（还有一些其他的杂项数据，不用怎么理会，只看原始数据就行）
  - `origin_comments_data.csv`：原始数据
  - `origin_videos_data.csv`：原始数据
- `Product_Recognition`：商品识别
  - `data`：存放数据
  - `Qwen_pred_data.csv`：预测结果
  - `Qwen_train.py`：预测主流程
- `Sentiment_Analysis`：情感分析
  - 运行顺序：data_clean，Bert_train.py划分数据，XFTranslator.py翻译增强，Bert_train.py，Bert_pred.py
  - `data`：存放数据
  - `model`：模型存放位置（[移动网盘链接](https://caiyun.139.com/w/i/2prAJSaEYDFq6)，提取码:b5ak）
  - `train_logs`：训练日志
  - `Bert_pred.py`：模型预测
  - `Bert_train.py`：模型训练
  - `data_clean.py`：数据清洗
  - `XFTranslator.py`：翻译增强
  - `EDA_util.py`：网上找的EDA实现
  - `FocalLoss.py`：网上找的FocalLoss实现
  - `machine_translation_python_demo.py`：官方API调用函数
- `Clustering_Insights`：聚类洞察
  - `data`：存放数据
  - `stop_words`：停用词表
  - `train.py`：聚类训练
  - `util.py`：工具包
- `baseline_改`：改动过的baseline（看看就好）
- `requirements.txt`：依赖文件
- `.gitignore`：忽略上传的文件
- `LICENSE`：开源许可证
