import random
from datetime import datetime
import time
from prefetch_generator import BackgroundGenerator
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, XLMRobertaModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import torch.nn.functional as F
# from util import *


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sentiment_classes_num = 5

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        sentiment_label, scenario_label, question_label, suggestion_label = self.labels[item]

        sentiment_label = int(sentiment_label)
        sentiment_one_hot = torch.zeros(self.sentiment_classes_num)
        sentiment_one_hot[sentiment_label - 1] = 1  # 将情感标签转换为one-hot编码

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
            'sentiment_label': sentiment_one_hot,  # 情感标签
            'scenario_label': torch.tensor(scenario_label, dtype=torch.float),  # 场景标签
            'question_label': torch.tensor(question_label, dtype=torch.float),  # 问题标签
            'suggestion_label': torch.tensor(suggestion_label, dtype=torch.float)  # 建议标签
        }


# './bert-base-multilingual-cased'
# './xlm-roberta-base'
# 'iic/nlp_polylm_qwen_7b_text_generation'
# 'xlm-roberta-large'
class MultiTaskBERT(nn.Module):
    def __init__(self, pretrained_model='./xlm-roberta-large', num_sentiment_labels=5):
        super().__init__()
        # 共享的BERT模型
        # self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert = XLMRobertaModel.from_pretrained(pretrained_model)
        # self.bert = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, device_map="cuda")
        # self.bert = AutoModelForCausalLM.from_pretrained(pretrained_model)


        # 情感分类 - 5分类
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_sentiment_labels)
        )

        self.scenario_head = self._build_binary_head()
        self.question_head = self._build_binary_head()
        self.suggestion_head = self._build_binary_head()

    # 场景、问题、建议分类 - 2分类 （共用一个结构，但不共享参数）
    def _build_binary_head(self):
        """辅助函数：构建独立的二分类头"""
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,  # (batch_size, sequence_length) 文本的token索引
            attention_mask=attention_mask,  # 注意力掩码，标记哪些位置是真实token 1或填充token 0
            return_dict=True  # 以字典形式返回
        )

        # [CLS] token ，(batch_size, hidden_size)，对所有批次（:）取第一个token（0，即[CLS]的位置）
        pooled_output = outputs.last_hidden_state[:, 0]

        return {
            'sentiment_logits': self.sentiment_head(pooled_output),
            'scenario_logits': self.scenario_head(pooled_output).squeeze(-1),  # squeeze(-1)：移除最后一维，变为 (batch_size,)
            'question_logits': self.question_head(pooled_output).squeeze(-1),
            'suggestion_logits': self.suggestion_head(pooled_output).squeeze(-1)
        }


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    创建一个计划，其学习率在优化器中设置的初始 lr 与 0 之间的余弦函数值后降低，
    在预热期期间，学习率在 0 和优化器中设置的初始 lr 之间线性增加。

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        为其调度学习率的优化器。
        num_warmup_steps (:obj:`int`):
        预热阶段的步骤数。
        num_training_steps (:obj:`int`):
        训练步骤的总数。
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        余弦 schedule 中的波数（默认值是在半余弦之后从最大值减少到 0）。
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        恢复训练时最后一个 epoch 的索引。

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # t/T 这个比值实际上是一个比例因子，
        # 当t=0（当前步数为0），学习率为0
        # 当t=num_warmup_steps（设定的预热步数），学习率达到设定的最大

        # decadence衰退阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # progress = (t-T)/(n-T)
        # 当t=T的时候，说明已经执行完预热步骤，progress=0
        # 当t=n的时候，说明训练要完成了，progress=1

        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
        # 当progress=0，返回1（比例因子），对应预热阶段结束，衰退阶段开始，最大学习率
        # 当progress=1，返回0（比例因子），对应衰退阶段结束，学习率为0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_step(train_data_loader,
               val_data_loader,
               model,
               optimizer,
               scheduler,
               device,
               valid_steps,
               total_steps,
               save_steps,
               sentiment_loss_fn,
               binary_loss_fn):
    writer = SummaryWriter(f"train_logs/BERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    min_val_loss = float('inf')
    train_iterator = iter(train_data_loader)

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_data_loader)
            batch = next(train_iterator)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        scenario_labels = batch['scenario_label'].to(device)
        question_labels = batch['question_label'].to(device)
        suggestion_labels = batch['suggestion_label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 计算情感分类损失
        sentiment_loss = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)

        # 计算其他任务的二元交叉熵损失
        scenario_loss = binary_loss_fn(outputs['scenario_logits'], scenario_labels)
        question_loss = binary_loss_fn(outputs['question_logits'], question_labels)
        suggestion_loss = binary_loss_fn(outputs['suggestion_logits'], suggestion_labels)

        # 总损失
        loss = sentiment_loss + scenario_loss + question_loss + suggestion_loss

        train_batch_loss = loss.item()

        writer.add_scalar('Loss/train', train_batch_loss, step)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], step)

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 更新进度条显示
        pbar.update()
        pbar.set_postfix(
            loss=f"{train_batch_loss:.2f}",
            step=step + 1,
        )

        if (step + 1) % valid_steps == 0:
            pbar.close()
            model.eval()
            val_losses = 0.0

            pbar = tqdm(total=len(val_data_loader.dataset), ncols=0, desc="Valid", unit=" uttr")

            for i, batch in enumerate(val_data_loader):
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    sentiment_labels = batch['sentiment_label'].to(device)
                    scenario_labels = batch['scenario_label'].to(device)
                    question_labels = batch['question_label'].to(device)
                    suggestion_labels = batch['suggestion_label'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    # 计算情感分类损失
                    sentiment_loss = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)

                    # 计算其他任务的二元交叉熵损失
                    scenario_loss = binary_loss_fn(outputs['scenario_logits'], scenario_labels)
                    question_loss = binary_loss_fn(outputs['question_logits'], question_labels)
                    suggestion_loss = binary_loss_fn(outputs['suggestion_logits'], suggestion_labels)

                    # 总损失
                    loss = sentiment_loss + scenario_loss + question_loss + suggestion_loss

                    val_losses += loss.item()

                pbar.update(val_data_loader.batch_size)
                pbar.set_postfix(
                    loss=f"{val_losses / (i + 1):.2f}",
                )

            pbar.close()
            model.train()

            avg_val_loss = val_losses / len(val_data_loader)
            writer.add_scalar('Loss/val', avg_val_loss, step)

            if avg_val_loss < min_val_loss:
                # 保存最佳模型
                torch.save(model, "model/best_val_loss_model.pt")
                min_val_loss = avg_val_loss
                print(f"save best_val_loss_model with val_loss: {min_val_loss:.4f}")

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # 定期保存模型
        if (step + 1) % save_steps == 0:
            torch.save(model, f"model/bert_model_step_{step + 1}.pt")
            print(f"Model saved at step {step + 1}")
    pbar.close()
    writer.close()
