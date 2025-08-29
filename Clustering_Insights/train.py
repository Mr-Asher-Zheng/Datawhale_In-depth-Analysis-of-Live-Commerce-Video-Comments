import time
from openai import OpenAI
from util import *

# video_data = pd.read_csv("./origin_data/origin_videos_data.csv")
# comments_data = pd.read_csv("./origin_data/origin_comments_data.csv")

# # 随机抽取10行查看
# print(video_data.sample(10))
# # 查看前5行
# print(comments_data.head())
#
# # 视频描述+视频标签
# video_data["text"] = video_data["video_desc"].fillna("") + " " + video_data["video_tags"].fillna("")
#
# # 检测语言
# video_data["language"] = video_data["text"].apply(safe_detect)
# print(video_data.sample(10))


# # 创建文本分类管道
# product_name_predictor = make_pipeline(
#     TfidfVectorizer(tokenizer=tokenizer, max_features=50, token_pattern=None),  # 中文分词 + TF-IDF特征提取
#     SGDClassifier()  # 随机梯度下降分类器
# )
#
# # 选择训练数据：从 video_data DataFrame 中筛选出 product_name 不为空的行，作为训练数据。
# product_name_predictor.fit(
#     video_data[~video_data["product_name"].isnull()]["text"],  # 非空文本作为特征
#     video_data[~video_data["product_name"].isnull()]["product_name"],  # 非空产品名作为标签
# )
#
# # 只对缺失product_name的数据进行商品名预测
# # df.loc[行选择条件, 列选择条件]
# video_data.loc[video_data["product_name"].isnull(), "product_name"] = product_name_predictor.predict(
#     video_data[video_data["product_name"].isnull()]["text"]
# )
#
# # 将预测数据保存到CSV文件
# video_data.to_csv("product_name_pred.csv", index=False)

# ================分界线=====================
# # 读取product_name_pred.csv文件
# video_data = pd.read_csv("Qwen_pred_data.csv")
#
# print(comments_data.columns)
#
# # 将comment_text中的emoji转换为文本
# comments_data["comment_text"] = comments_data["comment_text"].apply(emoji_to_text)
#
# # 处理杂项字符
# comments_data["comment_text"] = comments_data["comment_text"].apply(clean_char)
#
# # 定义要删除/修改的特定文本列表
# invalid_texts = ['$339 :0', '1']
# # # 使用 ~isin() 反向选择保留其他行
# # comments_data = comments_data[~comments_data['comment_text'].isin(invalid_texts)].copy()
# # 将这些文本替换为空字符串
# comments_data.loc[comments_data['comment_text'].isin(invalid_texts), 'comment_text'] = ''
#
# # 清除comment_text缺失的数据
# comments_data.dropna(subset=["comment_text"], inplace=True)
#
# # 检测评论语言为后面分词做准备
# comments_data["language"] = comments_data["comment_text"].apply(safe_detect)
#
# print(comments_data.iloc[0:10])  # 查看前10列数据
# # 输出语言分布
# print(comments_data["language"].value_counts())
#
# # 分类
# # 'sentiment_category',关于商品的情感倾向分类
# # 数值含义：1-正面，2-负面，3-正负都包含，4-中性，5-不相关。
#
# # 'user_scenario',是否与用户场景有关
# # 'user_question',是否与用户疑问有关
# # 'user_suggestion',是否与用户建议有关
# # 数值含义：1-相关，0-不相关
#
# for col in ['sentiment_category',
#             'user_scenario',
#             'user_question',
#             'user_suggestion']:
#     predictor = make_pipeline(
#         TfidfVectorizer(tokenizer=tokenizer, token_pattern=None),
#         SGDClassifier()
#     )
#     predictor.fit(
#         comments_data[~comments_data[col].isnull()]["comment_text"],
#         comments_data[~comments_data[col].isnull()][col],
#     )
#     comments_data[col] = predictor.predict(comments_data["comment_text"])

# ======================分界线=====================
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

# ====================================== 合并并保存数据 =====================================
video_data = pd.read_csv("../Product_Recognition/Qwen_pred_data.csv")
pred_comments_data = pd.read_csv("../Sentiment_Analysis/BERT_pred.csv")
origin_comments_data = pd.read_csv("../origin_data/origin_comments_data.csv")

# 将comments_data里的sentiment_category, user_scenario, user_question, user_suggestion列
# 根据video_id写入origin_comments_data里的sentiment_category, user_scenario, user_question, user_suggestion
# 先设置video_id为索引
origin_comments_data.set_index('video_id', inplace=True)
pred_comments_data.set_index('video_id', inplace=True)
# 只更新指定的列
origin_comments_data.update(
    pred_comments_data[['sentiment_category', 'user_scenario', 'user_question', 'user_suggestion']])
# 重置索引
origin_comments_data.reset_index(inplace=True)
# 保存为新 CSV
origin_comments_data.to_csv("./data/updated_comments_data.csv", index=False)
print("Updated comments data saved to 'updated_comments_data.csv'.")

# ======================================= 读取并清理数据 ====================================
# 读取新 CSV
comments_data = pd.read_csv("./data/updated_comments_data.csv")

# 转换特殊名词
# comments_data["comment_text"] = comments_data["comment_text"].apply(replace_custom_terms)
# 清除emoji
# comments_data["comment_text"] = comments_data["comment_text"].apply(remove_emoji)
# 清理文本
comments_data["comment_text"] = comments_data["comment_text"].apply(clean_text)


def global_stopwords():
    languages = stopwords.fileids()
    global_stop_words = set()
    for lang in languages:
        global_stop_words.update(stopwords.words(lang))

    json_files = ['./stop_words/en.json', './stop_words/zh.json', './stop_words/ja.json', ]
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            json_stopwords = set(json.load(f))
            global_stop_words.update(json_stopwords)

    myself_stop = {"w", "r", "f",
                   "ł", "ż", "ć", "ó",
                   "요", "중에", "이", "더", "어", "에",
                   "けど", "ね",
                   "ä", "ö", "ą", "ę"}
    global_stop_words.update(myself_stop)

    return global_stop_words


GLOBAL_STOPWORDS = global_stopwords()


def deal_stop_words(text):
    # words = jieba.lcut(text)
    words = spacy_tokenizer(text)
    words = [word for word in words if word.strip() and word not in GLOBAL_STOPWORDS]
    return ' '.join(words)


comments_data["comment_text"] = comments_data["comment_text"].apply(deal_stop_words)

# 恢复特殊名词
# comments_data["comment_text"] = comments_data["comment_text"].apply(restore_custom_terms)

# # 清除comment_text缺失的数据
# comments_data.dropna(subset=["comment_text"], inplace=True)

# comments_data.to_csv("./origin_data/embedding_comments_data.csv", index=False)
# print("saved to 'embedding_comments_data.csv'.")


# ========================================= 聚类 =========================================
# 'positive_cluster_theme',按正面倾向聚类的类簇主题词
# 'negative_cluster_theme',按负面倾向聚类的类簇主题词
# 'scenario_cluster_theme',按用户场景聚类的类簇主题词
# 'question_cluster_theme',按用户疑问聚类的类簇主题词
# 'suggestion_cluster_theme'按用户建议聚类的类簇主题词

# 每个聚类将提取前top_n_words=10个最具代表性的主题词
# 将数据分为n_clusters=8个簇


# model = XLMRobertaModel.from_pretrained('./xlm-roberta-large')
# tokenizer = XLMRobertaTokenizer.from_pretrained('./xlm-roberta-large')


# print(get_xlmr_embeddings("你好，hello, こんにちは"))
# time.sleep(60)


# def xlmr_tokenize(text):
#     # 使用XLMRoberta分词，并转换为词语列表
#     tokens = tokenizer.tokenize(text)
#     print(tokens)
#     return tokens

client = OpenAI(
    api_key="",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


def get_qwen_embeddings(texts, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(len(batch), batch)

        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=batch,
            dimensions=64,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )

        batch_embeddings = [item.embedding for item in completion.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def chose_best_clusters(data, cluster_theme_select, sentiment_category, name=None):
    # 千问大模型
    filtered_data = data[data[cluster_theme_select].isin(sentiment_category)]["comment_text"]
    filtered_data = filtered_data.tolist()

    embeddings = get_qwen_embeddings(filtered_data, batch_size=10)

    # 保存embedding到JSON文件
    with open(f'./data/{name}_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)
    print(f"Embeddings saved to {name}_embeddings.json")

    # 后续使用时加载
    with open(f'./data/{name}_embeddings.json', 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    # 预训练大模型
    # filtered_data = data[data[cluster_theme_select].isin(sentiment_category)]["comment_text"]
    # filtered_data = filtered_data.tolist()
    # embeddings = get_xlmr_embeddings(filtered_data)

    # 原始
    # filtered_data = data[data[cluster_theme_select].isin(sentiment_category)]["comment_text"]
    # tfidf_vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    # embeddings = tfidf_vectorizer.fit_transform(filtered_data)

    # 画分布图
    plot_embedding_distribution(embeddings, filtered_data)

    # print(embeddings)
    # time.sleep(60)

    best_k = 0
    best_score = -1

    k_range = range(5, 9)
    scores = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42,
            n_init=20,  # 增加初始化次数
            max_iter=500  # 增加最大迭代次数
        )
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Cluster Numbers')
    plt.grid(True)
    plt.show()

    return best_k


positive_n_clusters = chose_best_clusters(
    data=comments_data,
    cluster_theme_select='sentiment_category',
    sentiment_category=[1, 3],  # 正面,正负都包含
    name="positive"
)

negative_n_clusters = chose_best_clusters(
    data=comments_data,
    cluster_theme_select='sentiment_category',
    sentiment_category=[2, 3],  # 负面,正负都包含
    name="negative"
)
scenario_n_clusters = chose_best_clusters(
    data=comments_data,
    cluster_theme_select='user_scenario',
    sentiment_category=[1],  # 只考虑与用户场景相关的评论
    name="scenario"
)
question_n_clusters = chose_best_clusters(
    data=comments_data,
    cluster_theme_select='user_question',
    sentiment_category=[1],  # 只考虑与用户疑问相关的评论
    name="question"
)
suggestion_n_clusters = chose_best_clusters(
    data=comments_data,
    cluster_theme_select='user_suggestion',
    sentiment_category=[1],  # 只考虑与用户建议相关的评论
    name="suggestion"
)

print(f"正面聚类数: {positive_n_clusters}, "
      f"负面聚类数: {negative_n_clusters},"
      f"用户场景聚类数: {scenario_n_clusters}, "
      f"用户疑问聚类数: {question_n_clusters}, "
      f"用户建议聚类数: {suggestion_n_clusters}")

# embedding_data.to_csv("./origin_data/embedding_comments_data.csv", index=False)
# print("saved to 'embedding_comments_data.csv'.")
#
# time.sleep(60)


# 正面倾向聚类
comments_data, positive_topics = comment_clustering(
    data=comments_data,
    sentiment_category=[1, 3],  # 正面,正负都包含
    cluster_theme_select='sentiment_category',
    cluster_theme='positive_cluster_theme',
    n_clusters=positive_n_clusters,
    top_n_words=10,
    tokenizer=spacy_tokenizer,
    name="positive"
)

# 负面倾向聚类
comments_data, negative_topics = comment_clustering(
    data=comments_data,
    sentiment_category=[2, 3],  # 负面,正负都包含
    cluster_theme_select='sentiment_category',
    cluster_theme='negative_cluster_theme',
    n_clusters=negative_n_clusters,
    top_n_words=10,
    tokenizer=spacy_tokenizer,
    name="negative"
)

# 用户场景聚类
comments_data, scenario_topics = comment_clustering(
    data=comments_data,
    sentiment_category=[1],
    cluster_theme_select='user_scenario',
    cluster_theme='scenario_cluster_theme',
    n_clusters=scenario_n_clusters,
    top_n_words=10,
    tokenizer=spacy_tokenizer,
    name="scenario"
)

# 用户疑问聚类
comments_data, question_topics = comment_clustering(
    data=comments_data,
    sentiment_category=[1],
    cluster_theme_select='user_question',
    cluster_theme='question_cluster_theme',
    n_clusters=question_n_clusters,
    top_n_words=10,
    tokenizer=spacy_tokenizer,
    name="question"
)

# 用户建议聚类
comments_data, suggestion_topics = comment_clustering(
    data=comments_data,
    sentiment_category=[1],
    cluster_theme_select='user_suggestion',
    cluster_theme='suggestion_cluster_theme',
    n_clusters=suggestion_n_clusters,
    top_n_words=10,
    tokenizer=spacy_tokenizer,
    name="suggestion"
)

if not os.path.exists("submit"):
    os.makedirs("submit")

video_data[["video_id", "product_name"]].to_csv("submit/submit_videos.csv", index=None)
comments_data[['video_id', 'comment_id',
               'sentiment_category',
               'user_scenario',
               'user_question',
               'user_suggestion',
               'positive_cluster_theme',
               'negative_cluster_theme',
               'scenario_cluster_theme',
               'question_cluster_theme',
               'suggestion_cluster_theme']].to_csv("submit/submit_comments.csv", index=None)
