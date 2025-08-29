import html
import os
import re
import emoji
import pandas as pd
import unicodedata


def emoji_to_text(text):
    """
    将文本中的emoji表情转换为对应的文字描述
    参数:
        text (str): 包含emoji的原始文本
    返回:
        str: 将emoji转换为文字描述后的文本
    """
    # 使用emoji库的demojize函数将emoji转换为文字描述，并用空格作为分隔符
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 将下划线替换为空格，使文字描述更自然
    text = text.replace("_", " ")
    return text


def clean_text(text):
    text = unicodedata.normalize('NFKC', str(text))
    # 将 HTML/XML 实体编码（如 &#39; &amp;）转换为原始字符
    text = html.unescape(text)
    # 处理特定链接（保留文本内容）
    text = re.sub(r'<a\s+href=["\'][^>]+>([^<]+)<\/a>', r' \1', text)
    # 移除剩余HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 通用URL清除（可选，如果还需要处理裸URL）
    text = re.sub(r'https?:\/\/[^\s]+', 'link', text)
    # 标准化空白字符
    text = ' '.join(text.split())
    return text.strip()


# 读取原始评论数据
comments_data = pd.read_csv("../origin_data/origin_comments_data.csv")
# 清洗数据
comments_data["comment_text"] = comments_data["comment_text"].apply(emoji_to_text)
comments_data["comment_text"] = comments_data["comment_text"].apply(clean_text)

# 保存清洗后的数据
os.makedirs("./data", exist_ok=True)
comments_data.to_csv("./data/clean_comments_data.csv", index=False)
print("数据清洗完成")
