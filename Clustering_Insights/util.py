import html
import regex
import pandas as pd
import unicodedata
from langdetect import detect
import jieba
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import spacy
import emoji
import os
import time
from sklearn.metrics import silhouette_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import XLMRobertaModel
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use('TkAgg')
CUSTOM_TERMS = {
    "Xfaiyx Smart Recorder": "XFRS",
    "Xfaiyx Smart Translator": "XFST"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model = XLMRobertaModel.from_pretrained('./xlm-roberta-large').to(device)
# tokenizer = XLMRobertaTokenizer.from_pretrained('./xlm-roberta-large')

spacy_models = {
    "en": spacy.load("en_core_web_sm"),
    "ja": spacy.load("ja_core_news_sm"),
    "de": spacy.load("de_core_news_sm"),
    "ko": spacy.load("ko_core_news_sm"),
    "es": spacy.load("es_core_news_sm"),
    "zh-cn": spacy.load("zh_core_web_sm"),
    "zh-tw": spacy.load("zh_core_web_sm"),
    "it": spacy.load("it_core_news_sm"),
    "pl": spacy.load("pl_core_news_sm"),
    "default": spacy.load("xx_ent_wiki_sm")
}


def safe_detect(text):
    supported_languages = ["en", "ja", "de", "ko", "es", "zh-cn", "zh-tw", "it", "pl"]

    if text.strip():  # 检查文本是否非空
        language = detect(text)
        return language if language in supported_languages else "default"
    else:
        return "default"


# 转换特殊名词
def replace_custom_terms(text):
    for term, replacement in CUSTOM_TERMS.items():
        text = regex.sub(rf'\b{regex.escape(term)}\b', replacement, text, flags=regex.IGNORECASE)
    return text


# 恢复特殊名词
def restore_custom_terms(text):
    for term, replacement in CUSTOM_TERMS.items():
        text = text.replace(replacement, term)
    return text


# 根据不同的语言选择不同的分词器
def spacy_tokenizer(text):
    try:
        language = safe_detect(text)
        if language in spacy_models:
            # 从 spacy_models 中获取对应语言的模型
            spacy_m = spacy_models[language]
            # for term in CUSTOM_TERMS.values():
            #     spacy_m.tokenizer.add_special_case(term, [{"ORTH": term}])
            # 使用 spacy 进行分词
            doc = spacy_m(text)
            # 返回分词后的文本
            return [token.text for token in doc]
    except Exception as e:
        print(f"⚠️ 分词错误: {e}，文本: '{text}'")
        return text.split()


def clean_text(text):
    text = unicodedata.normalize('NFKC', str(text))
    # 将 HTML/XML 实体编码（如 &#39; &amp;）转换为原始字符
    text = html.unescape(text)
    # 处理特定链接（保留文本内容）
    text = regex.sub(r'<a\s+href=["\'][^>]+>([^<]+)<\/a>', r' \1', text)
    # 移除剩余HTML标签
    text = regex.sub(r'<[^>]+>', ' ', text)
    # 通用URL清除（可选，如果还需要处理裸URL）
    text = regex.sub(r'https?:\/\/[^\s]+', 'link', text)
    # 去除特殊符号
    text = regex.sub(r'[^\p{L}\p{M}\p{Emoji}\s]|[\d]', '', text)
    # 标准化空白字符
    text = ' '.join(text.split())
    # 文本转小写
    text = text.lower()
    return text.strip()


def remove_emoji(text):
    if isinstance(text, str):  # 确保是字符串
        return emoji.replace_emoji(text, replace="")  # 替换 emoji 为空
    return text  # 如果不是字符串（如 NaN），直接返回


def get_xlmr_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def comment_clustering(data, sentiment_category, cluster_theme_select, cluster_theme, n_clusters=8, top_n_words=10,
                       tokenizer=jieba.lcut, name=None):
    """
    对评论进行聚类分析，提取主题词。
    :param data: DataFrame，评论数据

    :param sentiment_category: list，
    指定情感或场景类别的列表，
    情感类别[1, 2, 3, 4, 5]分别代表正面、负面、正负都包含、中性、不相关
    场景类别[1，0]分别代表与用户场景相关和不相关

    :param cluster_theme_select: str，
    选择用于聚类的情感或场景类别列名，
    情感类别：'sentiment_category'
    场景类别：'user_scenario', 'user_question', 'user_suggestion'

    :param cluster_theme: str，聚类主题词存储的列名
    :param n_clusters: int，聚类的数量，默认为8
    :param top_n_words: int，每个聚类提取的主题词数量，默认为10
    :param tokenizer: function，用于分词的函数，默认为jieba.lcut

    :return: DataFrame，包含聚类主题词的新列
    :return: list，包含每个聚类的主题词
    """
    filtered_data = data[data[cluster_theme_select].isin(sentiment_category)]
    texts = filtered_data["comment_text"].tolist()

    # embeddings = get_xlmr_embeddings(texts)

    # 后续使用时加载
    with open(f'./data/{name}_embeddings.json', 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',  # 增加初始化次数
        random_state=42,
        n_init=20,  # 增加初始化次数
        max_iter=500  # 增加最大迭代次数
    )
    kmeans_cluster_label = kmeans.fit_predict(embeddings)
    print("实际迭代次数:", kmeans.n_iter_)
    # print(f"聚类标签: {kmeans_cluster_label}")

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=min(500, len(texts) * 10),
        min_df=1,
        # ngram_range=(1, 2),  # 捕获短语
        use_idf=True,  # 启用IDF权重
        smooth_idf=True,  # 避免除零错误
        sublinear_tf=True,  # 对数频率缩放
        norm='l2',
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    # 获取TF-IDF向量化后的所有特征词(词汇表)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # print(f"TF-IDF特征词数量: {len(feature_names)}")
    # print(f"TF-IDF特征词示例: {feature_names[:10]}")

    kmeans_top_word = []
    # 遍历所有聚类
    for i in range(n_clusters):
        # 获取当前聚类的文档索引
        cluster_indices = np.where(kmeans_cluster_label == i)[0]
        # print(f"聚类 {i} 的文档索引: {cluster_indices}")
        # 提取当前聚类的TF-IDF矩阵
        cluster_tfidf = tfidf_matrix[cluster_indices]
        # print(f"聚类 {i} 的TF-IDF矩阵形状: {cluster_tfidf.shape}")
        # print(f"聚类 {i} 的TF-IDF矩阵示例: {cluster_tfidf[:5].toarray()}")

        # 计算当前聚类的平均TF-IDF向量
        avg_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        # 获取最重要的top_n_words个词的索引
        top_indices = avg_tfidf.argsort()[-top_n_words:][::-1]
        # 将索引映射为实际词语
        top_words = ' '.join([feature_names[idx] for idx in top_indices])
        kmeans_top_word.append(top_words)

    # 将聚类结果写回原数据
    data.loc[data[cluster_theme_select].isin(sentiment_category), cluster_theme] = [kmeans_top_word[x] for x in
                                                                                    kmeans_cluster_label]

    print(f"主题词{kmeans_top_word}")
    # time.sleep(60)
    return data, kmeans_top_word

    # ============================================================================

    # filtered_data = data[data[cluster_theme_select].isin(sentiment_category)]
    #
    # kmeans_predictor = make_pipeline(
    #     TfidfVectorizer(tokenizer=tokenizer), KMeans(n_clusters=n_clusters)
    # )
    #
    # # 筛选指定情感类别的评论进行聚类训练，例如sentiment_category=[1,3]
    # kmeans_predictor.fit(filtered_data["comment_text"])
    #
    # # 预测聚类标签
    # kmeans_cluster_label = kmeans_predictor.predict(filtered_data["comment_text"])
    #
    # # 用于存储每个聚类的主题词
    # kmeans_top_word = []
    #
    # # 获取TF-IDF向量化器和KMeans模型
    # tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
    # kmeans_model = kmeans_predictor.named_steps['kmeans']
    #
    # # 获取TF-IDF向量化后的所有特征词(词汇表)
    # feature_names = tfidf_vectorizer.get_feature_names_out()
    # # 获取K-means模型中每个聚类的中心点坐标
    # cluster_centers = kmeans_model.cluster_centers_
    #
    # # 对每个聚类n_clusters=8
    # for i in range(kmeans_model.n_clusters):
    #     # 将聚类中心向量的值按降序排列，获取索引
    #     top_feature_indices = cluster_centers[i].argsort()[::-1]
    #     # 取前top_n_words(10)个索引对应的特征词
    #     top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    #     kmeans_top_word.append(top_word)
    #
    # # 只针对情感类别为1(正面)或3(负面)的评论：
    # # 根据之前预测的聚类标签kmeans_cluster_label，查找对应的主题词
    # # 将主题词存入新列positive_cluster_theme中
    # data.loc[data[cluster_theme_select].isin(sentiment_category), cluster_theme] = [kmeans_top_word[x] for x in
    #                                                                                 kmeans_cluster_label]
    # print(f"主题词{kmeans_top_word}")
    # return data, kmeans_top_word


def plot_embedding_distribution(embeddings, texts):
    """
    可视化嵌入向量的分布。
    :param embeddings: numpy array，嵌入向量
    :param texts: list，原始文本
    """
    embeddings = np.array(embeddings)
    n_samples = embeddings.shape  # 获取样本数量
    print(f"样本数量: {n_samples}, 嵌入维度: {embeddings.shape[1]}")

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=min(30, len(texts) - 1))

    pca_result = pca.fit_transform(embeddings)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.6)
    plt.title('PCA of Embeddings')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.subplot(1, 2, 2)
    plt.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], alpha=0.6)
    plt.title('t-SNE of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()
