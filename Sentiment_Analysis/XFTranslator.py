import pandas as pd
from machine_translation_python_demo import *
from langdetect import detect, LangDetectException

# 讯飞API
APPId = ""
APISecret = ""
APIKey = ""


def translate_text(text, from_lang, to_lang):
    """使用讯飞API翻译文本"""
    url = 'https://itrans.xf-yun.com/v1/its'

    body = {
        "header": {
            "app_id": APPId,
            "status": 3,
            # "res_id": RES_ID
        },
        "parameter": {
            "its": {
                "from": from_lang,
                "to": to_lang,
                "result": {}
            }
        },
        "payload": {
            "input_data": {
                "encoding": "utf8",
                "status": 3,
                "text": base64.b64encode(text.encode("utf-8")).decode('utf-8')
            }
        }
    }

    request_url = assemble_ws_auth_url(url, "POST", APIKey, APISecret)
    headers = {'content-type': "application/json", 'host': 'itrans.xf-yun.com', 'app_id': APPId}

    try:
        response = requests.post(request_url, data=json.dumps(body), headers=headers)
        tempResult = json.loads(response.content.decode())
        # translated_text = base64.b64decode(tempResult['payload']['result']['text']).decode()
        translated_text = tempResult['payload']['result']['text']
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        return None


def detect_language(text):
    """检测文本语言"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return None


def translate_and_augment(texts, labels):
    """翻译增强功能"""
    translated_texts = []
    translated_labels = []

    for text, label in zip(texts, labels):
        # 检测语言
        try:
            lang = detect_language(text)
            if lang is None:
                print(f"无法检测语言，跳过: {text}")
                continue

            # 根据语言决定翻译方向
            if lang == 'en':
                # 英文翻译成中文
                translated = translate_text(text, 'en', 'cn')
                if translated:
                    translated_texts.append(translated)
                    translated_labels.append(label)
            else:
                # 其他语言翻译成英文
                translated = translate_text(text, lang, 'en')
                if translated:
                    translated_texts.append(translated)
                    translated_labels.append(label)

        except Exception as e:
            print(f"处理文本时出错: {text}, 错误: {e}")
            continue

    return translated_texts, translated_labels


# if __name__ == "__main__":
#     # 读取CSV文件
#     train_df = pd.read_csv("./origin_data/train_comments_data.csv")
#     train_texts = train_df["comment_text"].tolist()
#     train_labels = train_df["multi_labels"].apply(eval).tolist()
#
#     # 执行翻译增强
#     translated_texts, translated_labels = translate_and_augment(train_texts, train_labels)
#     train_texts.extend(translated_texts)
#     train_labels.extend(translated_labels)
#
#     # 翻译后保存
#     train_df = pd.DataFrame({
#         'comment_text': train_texts,
#         'multi_labels': train_labels
#     })
#
#     train_df.to_csv("./data/translate_train_comments_data.csv", index=False)
