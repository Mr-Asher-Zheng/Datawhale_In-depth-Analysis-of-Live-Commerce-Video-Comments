import time
import pandas as pd
import torch
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
# from Bert_util import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info]: Use {device} now!")

comments_data = pd.read_csv("./data/clean_comments_data.csv")
# 原来数据
filtered_data = comments_data[~comments_data["sentiment_category"].isnull()].copy()
# 需要预测的评论数据
pred_comments_data = comments_data[comments_data["sentiment_category"].isnull()].copy()

pred_texts = pred_comments_data["comment_text"].tolist()
# tokenizer = BertTokenizer.from_pretrained("./bert-base-multilingual-cased")
# tokenizer = XLMRobertaTokenizer.from_pretrained("./xlm-roberta-base")
tokenizer = XLMRobertaTokenizer.from_pretrained("./xlm-roberta-large")


class PredDataset(Dataset):
    def __init__(self, tests, tokenizer, max_len):
        self.tests = tests
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tests)

    def __getitem__(self, item):
        text = str(self.tests[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 是否添加特殊token,开头[CLS]和结尾[SEP]
            max_length=self.max_len,  # 设置编码后的最大长度（包括特殊token）
            return_token_type_ids=False,  # 是否返回token类型ID，单句任务通常不需要，对于多语言模型，有时需要设为True
            padding='max_length',  # 填充[PAD]，'max_length'：填充到max_length指定长度，'longest'：填充到批次中最长序列，False或'do_not_pad'：不填充
            truncation=True,  # 是否截断超过max_length的文本
            return_attention_mask=True,  # attention mask用于区分真实token和padding token，1表示真实token，0表示padding
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),  # 输入ID，形状为(batch_size, max_len)
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力掩码，形状为(batch_size, max_len)
        }


pred_dataset = PredDataset(pred_texts, tokenizer, max_len=512)

dataloader = DataLoader(
    pred_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

# model_path = "./model/best_val_loss_model.pt"
model_path = "./model/best_f1_model.pt"

model = torch.load(model_path, weights_only=False)
model.to(device)
model.eval()  # 将模型设置为评估模式

# 预测过程
sentiment_preds = []
scenario_preds = []
question_preds = []
suggestion_preds = []

# 预测
for i, batch in enumerate(dataloader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 获取预测结果
        sentiment = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy() + 1  # 转换为1-5
        scenario = (torch.sigmoid(outputs['scenario_logits']) > 0.5).int().cpu().numpy()
        question = (torch.sigmoid(outputs['question_logits']) > 0.5).int().cpu().numpy()
        suggestion = (torch.sigmoid(outputs['suggestion_logits']) > 0.5).int().cpu().numpy()

        sentiment_preds.extend(sentiment)
        scenario_preds.extend(scenario)
        question_preds.extend(question)
        suggestion_preds.extend(suggestion)

# 将预测结果写在pred_comments_data
# 将预测结果添加到预测数据中
pred_comments_data['sentiment_category'] = sentiment_preds
pred_comments_data['user_scenario'] = scenario_preds
pred_comments_data['user_question'] = question_preds
pred_comments_data['user_suggestion'] = suggestion_preds

# 和原来有的数据做拼接
final_data = pd.concat([filtered_data, pred_comments_data], axis=0)

# 保存结果
output_path = "./BERT_pred.csv"
final_data.to_csv(output_path, index=False)
print(f"[Info]: Prediction completed and saved to {output_path}")
