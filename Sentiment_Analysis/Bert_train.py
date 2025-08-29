from datetime import datetime
import time
from prefetch_generator import BackgroundGenerator
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import torch.nn.functional as F
from EDA_util import *
from XFTranslator import *


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

        # 无标签平滑
        # sentiment_label = torch.tensor(int(sentiment_label) - 1)
        # sentiment_one_hot = torch.zeros(self.sentiment_classes_num)
        # sentiment_one_hot[sentiment_label - 1] = 1  # 将情感标签转换为one-hot编码

        sentiment_label = torch.tensor(int(sentiment_label) - 1)
        epsilon = 0.1
        sentiment_one_hot = F.one_hot(sentiment_label, self.sentiment_classes_num)  # 转换成one-hot
        sentiment_one_hot = torch.clamp(sentiment_one_hot.float(), min=epsilon / (self.sentiment_classes_num - 1),
                                        max=1.0 - epsilon)

        # # 有标签平滑
        # sentiment_label = int(sentiment_label)
        # epsilon = 0.1
        # sentiment_one_hot = torch.full(
        #     (self.sentiment_classes_num,),
        #     epsilon / (self.sentiment_classes_num - 1)
        # )
        # sentiment_one_hot[sentiment_label - 1] = 1 - epsilon

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


def train_step():
    writer = SummaryWriter(f"./train_logs/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    min_val_loss = float('inf')
    max_total_F1 = 0.0
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
        # 交叉熵损失
        sentiment_loss = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)

        # 计算其他任务的二元交叉熵损失
        scenario_loss = scenario_binary_loss_fn(outputs['scenario_logits'], scenario_labels)
        question_loss = question_binary_loss_fn(outputs['question_logits'], question_labels)
        suggestion_loss = suggestion_binary_loss_fn(outputs['suggestion_logits'], suggestion_labels)

        # 总损失
        loss = sentiment_loss + scenario_loss + question_loss + suggestion_loss

        # # 分别反向传播每个任务的损失
        # # 保留计算图用于后续反向传播
        # retain_graph = True
        #
        # # 1. 情感任务反向传播
        # sentiment_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.sentiment_head.parameters(), 1.0)
        # optimizers["sentiment"].step()
        # optimizers["sentiment"].zero_grad()
        # schedulers["sentiment"].step()
        #
        # # 2. 场景任务反向传播
        # scenario_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.scenario_head.parameters(), 1.0)
        # optimizers["scenario"].step()
        # optimizers["scenario"].zero_grad()
        # schedulers["scenario"].step()
        #
        # # 3. 问题任务反向传播
        # question_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.question_head.parameters(), 1.0)
        # optimizers["question"].step()
        # optimizers["question"].zero_grad()
        # schedulers["question"].step()
        #
        # # 4. 建议任务反向传播
        # suggestion_loss.backward(retain_graph=False)
        # torch.nn.utils.clip_grad_norm_(model.suggestion_head.parameters(), 1.0)
        # optimizers["suggestion"].step()
        # optimizers["suggestion"].zero_grad()
        # schedulers["suggestion"].step()
        #
        # # 5. 最后更新BERT共享层
        # # 需要重新计算BERT的梯度（因为之前只计算了头部的梯度）
        # torch.nn.utils.clip_grad_norm_(model.bert.parameters(), 1.0)
        # optimizers["bert"].step()
        # optimizers["bert"].zero_grad()
        # schedulers["bert"].step()
        #
        # # train_batch_loss = loss.item()

        train_batch_loss = sentiment_loss.item() + scenario_loss.item() + question_loss.item() + suggestion_loss.item()

        writer.add_scalar('Loss/train', train_batch_loss, step)
        # writer.add_scalar('Learning Rate', optimizers["bert"].param_groups[0]['lr'], step)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], step)

        # 反向传播和参数更新
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

            val_f1 = {
                'sentiment': 0.0,
                'scenario': 0.0,
                'question': 0.0,
                'suggestion': 0.0
            }

            task_weights = {
                'sentiment': 0.4,  # 情感分析更重要
                'scenario': 0.20,
                'question': 0.20,
                'suggestion': 0.20
            }

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
                    # 交叉熵损失
                    sentiment_loss = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)

                    # 计算其他任务的二元交叉熵损失
                    scenario_loss = scenario_binary_loss_fn(outputs['scenario_logits'], scenario_labels)
                    question_loss = question_binary_loss_fn(outputs['question_logits'], question_labels)
                    suggestion_loss = suggestion_binary_loss_fn(outputs['suggestion_logits'], suggestion_labels)

                    # 总损失
                    loss = sentiment_loss + scenario_loss + question_loss + suggestion_loss

                    val_losses += loss.item()

                    # 计算F1分数
                    sentiment_f1 = calculate_F1(outputs['sentiment_logits'], sentiment_labels, type='multiclass')['f1']
                    scenario_f1 = calculate_F1(outputs['scenario_logits'], scenario_labels, type='binary')['f1']
                    question_f1 = calculate_F1(outputs['question_logits'], question_labels, type='binary')['f1']
                    suggestion_f1 = calculate_F1(outputs['suggestion_logits'], suggestion_labels, type='binary')['f1']

                    val_f1['sentiment'] += sentiment_f1
                    val_f1['scenario'] += scenario_f1
                    val_f1['question'] += question_f1
                    val_f1['suggestion'] += suggestion_f1

                pbar.update(val_data_loader.batch_size)
                pbar.set_postfix(
                    loss=f"{val_losses / (i + 1):.2f}",
                )

            pbar.close()
            model.train()

            avg_val_loss = val_losses / len(val_data_loader)
            # 计算平均F1分数
            avg_sentiment_f1 = val_f1['sentiment'] / len(val_data_loader)
            avg_scenario_f1 = val_f1['scenario'] / len(val_data_loader)
            avg_question_f1 = val_f1['question'] / len(val_data_loader)
            avg_suggestion_f1 = val_f1['suggestion'] / len(val_data_loader)

            avg_total_F1 = (
                    avg_sentiment_f1 * task_weights['sentiment'] +
                    avg_scenario_f1 * task_weights['scenario'] +
                    avg_question_f1 * task_weights['question'] +
                    avg_suggestion_f1 * task_weights['suggestion']
            )

            # 记录损失值
            writer.add_scalar('Loss/val', avg_val_loss, step)
            # 记录F1值
            writer.add_scalar('F1/sentiment', avg_sentiment_f1, step)
            writer.add_scalar('F1/scenario', avg_scenario_f1, step)
            writer.add_scalar('F1/question', avg_question_f1, step)
            writer.add_scalar('F1/suggestion', avg_suggestion_f1, step)
            writer.add_scalar('F1/total', avg_total_F1, step)

            if avg_val_loss < min_val_loss:
                # 保存验证损失最小模型
                torch.save(model, "./model/best_val_loss_model.pt")
                min_val_loss = avg_val_loss
                print(f"save best_val_loss_model with val_loss: {min_val_loss:.4f}")

            if avg_total_F1 > max_total_F1:
                # 保存F1分数最高模型
                torch.save(model, "./model/best_f1_model.pt")
                max_total_F1 = avg_total_F1
                print(f"save best_f1_model with avg_total_F1: {avg_total_F1:.4f}")

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # 定期保存模型
        if (step + 1) % save_steps == 0:
            torch.save(model, f"./model/bert_model_step_{step + 1}.pt")
            print(f"Model saved at step {step + 1}")
    pbar.close()
    writer.close()


def calculate_F1(preds, labels, type):
    if type == 'binary':
        preds = (torch.sigmoid(preds) > 0.5).int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        f1 = f1_score(labels, preds, average='binary')
    elif type == 'multiclass':
        preds = torch.argmax(preds, dim=1).cpu().numpy() + 1
        labels = torch.argmax(labels, dim=1).cpu().numpy() + 1
        # labels = labels.int().cpu().numpy() + 1  # 转换为1-5
        f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}


def augment_text_data(texts, labels, alpha_sr=0.05, alpha_ri=0.05, alpha_rs=0.05, p_rd=0.05, num_aug=12):
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        # 进行EDA数据增强
        new_text = eda(
            text,
            alpha_sr=alpha_sr,
            alpha_ri=alpha_ri,
            alpha_rs=alpha_rs,
            p_rd=p_rd,
            num_aug=num_aug
        )
        for aug_text in new_text[1:]:
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', is_multiclass=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.is_multiclass = is_multiclass

    def forward(self, inputs, targets):
        if self.is_multiclass:

            targets_one_hot = targets
            softmax = torch.softmax(inputs, dim=1)
            log_softmax = torch.log(softmax)
            ce_loss = - (targets_one_hot * log_softmax).sum(dim=1)

            p_t = (softmax * targets_one_hot).sum(dim=1)
            focal_weight = (1 - p_t) ** self.gamma

            # 如果指定了alpha，应用类别权重
            if self.alpha is not None:
                targets_idx = targets_one_hot.argmax(dim=1)
                alpha_t = self.alpha.gather(0, targets_idx)
                focal_weight = alpha_t * focal_weight

            # 计算最终的focal loss
            focal_loss = focal_weight * ce_loss

            # # 多分类情况
            # targets_one_hot = targets
            #
            # # 计算交叉熵项
            # targets_idx = targets_one_hot.argmax(dim=1)
            # ce_loss = F.cross_entropy(inputs, targets_idx, reduction='none')
            #
            # # 计算softmax概率
            # probs = F.softmax(inputs, dim=1)
            #
            # # 防止数值不稳定
            # probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
            #
            # # 计算pt
            # p_t = (probs * targets_one_hot).sum(dim=1)
            # # focal weight权重
            # focal_weight = (1 - p_t) ** self.gamma
            #
            # # 如果指定了alpha，应用类别权重
            # if self.alpha is not None:
            #     alpha_t = self.alpha.gather(0, targets_idx)
            #     focal_weight = alpha_t * focal_weight
            #
            # # 计算最终的focal loss
            # focal_loss = focal_weight * ce_loss

        else:
            # 二分类情况
            # 计算sigmoid激活，得到正类的预测概率
            probs = torch.sigmoid(inputs)

            # 防止数值不稳定
            probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

            # 计算交叉熵项
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

            # 计算focal weight
            # 对于正类(target=1)，weight = (1 - p)^gamma
            # 对于负类(target=0)，weight = p^gamma
            p_t = targets * probs + (1 - targets) * (1 - probs)
            focal_weight = (1 - p_t) ** self.gamma

            # 如果指定了alpha，应用类别权重
            if self.alpha is not None:
                alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
                focal_weight = alpha_t * focal_weight

            # 计算最终的focal loss
            focal_loss = focal_weight * bce_loss

        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    setup_seed(42)
    os.makedirs("./model", exist_ok=True)

    comments_data = pd.read_csv("./data/clean_comments_data.csv")

    # ================================================================================
    # 过滤空数据并划分数据
    filtered_data = comments_data[~comments_data["sentiment_category"].isnull()].copy()

    # # 统计情感类别、场景、问题和建议的分布
    # sentiment_counts = filtered_data["sentiment_category"].value_counts()
    # scenario_counts = filtered_data["user_scenario"].value_counts()
    # question_counts = filtered_data["user_question"].value_counts()
    # suggestion_counts = filtered_data["user_suggestion"].value_counts()
    # print("情感类别分布：")
    # print(sentiment_counts)
    # print("用户场景分布：")
    # print(scenario_counts)
    # print("用户问题分布：")
    # print(question_counts)
    # print("用户建议分布：")
    # print(suggestion_counts)
    #
    # time.sleep(60)

    texts = filtered_data["comment_text"].tolist()
    sentiment_labels = filtered_data["sentiment_category"].tolist()
    scenario_labels = filtered_data["user_scenario"].tolist()
    question_labels = filtered_data["user_question"].tolist()
    suggestion_labels = filtered_data["user_suggestion"].tolist()

    multi_labels = list(zip(sentiment_labels, scenario_labels, question_labels, suggestion_labels))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        multi_labels,
        test_size=0.3,
        random_state=42
    )
    # print(train_texts)
    print(train_labels)
    # 将训练集和验证集存放到新的csv文件
    train_df = pd.DataFrame({
        'comment_text': train_texts,
        'multi_labels': train_labels
    })
    val_df = pd.DataFrame({
        'comment_text': val_texts,
        'multi_labels': val_labels
    })
    train_df.to_csv("./data/train_comments_data.csv", index=False)
    val_df.to_csv("./data/val_comments_data.csv", index=False)
    print("Train and Val datasets created successfully.")

    # # ================================================================================
    # # 翻译增强，这一部分是XFTranslator.py的main代码，可以在运行完上面代码后，在XFTranslator.py分开执行，不然每次重新训练都要消耗讯飞的token
    # # 读取CSV文件
    # train_df = pd.read_csv("./data/train_comments_data.csv")
    # train_texts = train_df["comment_text"].tolist()
    # train_labels = train_df["multi_labels"].apply(eval).tolist()
    #
    # # 执行翻译增强
    # translated_texts, translated_labels = translate_and_augment(train_texts, train_labels)
    # train_texts.extend(translated_texts)
    # train_labels.extend(translated_labels)
    #
    # # 翻译后保存
    # train_df = pd.DataFrame({
    #     'comment_text': train_texts,
    #     'multi_labels': train_labels
    # })
    #
    # train_df.to_csv("./data/translate_train_comments_data.csv", index=False)
    # # ================================================================================

    # 读取训练集
    train_df = pd.read_csv("./data/translate_train_comments_data.csv")
    train_texts = train_df["comment_text"].tolist()
    train_labels = train_df["multi_labels"].apply(eval).tolist()
    # 读取验证集
    val_df = pd.read_csv("./data/val_comments_data.csv")
    val_texts = val_df["comment_text"].tolist()
    val_labels = val_df["multi_labels"].apply(eval).tolist()

    # 翻译增强后
    print("After back-translation augmentation:", len(train_texts))

    # EDA增强前
    print("Before EDA augmentation:", len(train_texts))

    new_train_texts, new_train_labels = augment_text_data(
        train_texts,
        train_labels,
        alpha_sr=0.05,  # synonym replacement
        alpha_ri=0.1,  # random insertion
        alpha_rs=0.1,  # random swap
        # p_rd=0.05,  # random deletion
        num_aug=2  # number of augmented sentences per original sentence
    )
    train_texts.extend(new_train_texts)
    train_labels.extend(new_train_labels)

    print("After EDA augmentation:", len(train_texts))

    # # 统计情感类别、场景、问题和建议的分布
    # sentiment_counts = pd.Series([label[0] for label in train_labels]).value_counts()
    # scenario_counts = pd.Series([label[1] for label in train_labels]).value_counts()
    # question_counts = pd.Series([label[2] for label in train_labels]).value_counts()
    # suggestion_counts = pd.Series([label[3] for label in train_labels]).value_counts()
    # print("训练集情感类别分布：")
    # print(sentiment_counts)
    # print("训练集用户场景分布：")
    # print(scenario_counts)
    # print("训练集用户问题分布：")
    # print(question_counts)
    # print("训练集用户建议分布：")
    # print(suggestion_counts)
    #
    # time.sleep(60)

    max_len = 64
    batch_size = 32
    total_steps = 2000
    warmup_steps = int(0.1 * total_steps)  # 10% = 100
    # valid_steps = int(0.25 * total_steps)  # 25% = 500
    save_steps = 500
    valid_steps = 25

    # tokenizer = BertTokenizer.from_pretrained("./bert-base-multilingual-cased")
    tokenizer = XLMRobertaTokenizer.from_pretrained("./xlm-roberta-large")

    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, max_len)

    train_data_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True  # 保持worker进程
    )

    val_data_loader = DataLoaderX(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True  # 保持worker进程
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MultiTaskBERT()
    model = model.to(device)

    # ================================= optimizer =======================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)

    # optimizers = {
    #     'bert': torch.optim.AdamW(model.bert.parameters(), lr=2e-5, weight_decay=1e-5),
    #     'sentiment': torch.optim.AdamW(model.sentiment_head.parameters(), lr=2e-5, weight_decay=1e-5),
    #     'scenario': torch.optim.AdamW(model.scenario_head.parameters(), lr=2e-5, weight_decay=1e-5),
    #     'question': torch.optim.AdamW(model.question_head.parameters(), lr=2e-5, weight_decay=1e-5),
    #     'suggestion': torch.optim.AdamW(model.suggestion_head.parameters(), lr=2e-5, weight_decay=1e-5)
    # }

    # ================================= 损失函数 =======================================

    # 情感分类使用交叉熵损失(标签用1~5的时候用)
    # sentiment_loss_fn = nn.CrossEntropyLoss().to(device)
    # BCEWithLogitsLoss

    # # 情感分类使用交叉熵损失(标签用one-hot编码的时候用)
    # # 计算并添加损失权重
    # sentiment_counts = [1552, 1691, 940, 2006, 9057]  # 1.0到5.0的样本数，一个样本对应一个值
    # sentiment_weights = 1. / torch.tensor(sentiment_counts, dtype=torch.float)
    # sentiment_weights = sentiment_weights / sentiment_weights.sum() * len(sentiment_counts)  # 归一化
    # sentiment_weights = sentiment_weights.to(device)
    # sentiment_loss_fn = nn.CrossEntropyLoss(
    #     weight=sentiment_weights,
    # )

    # 使用Focal Loss
    sentiment_counts = [564, 649, 398, 762, 3179]  # 1.0到5.0的样本数
    sentiment_weights = 1. / torch.tensor(sentiment_counts, dtype=torch.float)
    # sentiment_weights = sentiment_weights / sentiment_weights.sum() * len(sentiment_counts)  # 归一化

    sentiment_weights = sentiment_weights / sentiment_weights.max()
    sentiment_weights = sentiment_weights.to(device)
    sentiment_loss_fn = FocalLoss(
        alpha=sentiment_weights,
        gamma=2.0,
        is_multiclass=True  # 多分类
    )

    # 其他任务使用Focal Loss
    # 计算每个任务的正类权重（基于样本比例的倒数）
    scenario_pos_weight = 4860 / 692  # 场景分类: 负类13551 vs 正类1695
    question_pos_weight = 4767 / 785  # 问题分类: 负类13201 vs 正类2045
    suggestion_pos_weight = 5144 / 408  # 建议分类: 负类14279 vs 正类967

    # 归一化权重，使正负类权重之和为1
    scenario_alpha = scenario_pos_weight / (1 + scenario_pos_weight)
    question_alpha = question_pos_weight / (1 + question_pos_weight)
    suggestion_alpha = suggestion_pos_weight / (1 + suggestion_pos_weight)

    # 为每个任务创建带类别权重的Focal Loss
    # α=正类权重
    scenario_binary_loss_fn = FocalLoss(alpha=scenario_alpha, gamma=2.0, is_multiclass=False)
    question_binary_loss_fn = FocalLoss(alpha=question_alpha, gamma=2.0, is_multiclass=False)
    suggestion_binary_loss_fn = FocalLoss(alpha=suggestion_alpha, gamma=2.0, is_multiclass=False)

    # ================================= 学习率调度器 =======================================

    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )
    # schedulers = {
    #     'bert': get_cosine_schedule_with_warmup(optimizers['bert'], warmup_steps, total_steps),
    #     'sentiment': get_cosine_schedule_with_warmup(optimizers['sentiment'], warmup_steps, total_steps),
    #     'scenario': get_cosine_schedule_with_warmup(optimizers['scenario'], warmup_steps, total_steps),
    #     'question': get_cosine_schedule_with_warmup(optimizers['question'], warmup_steps, total_steps),
    #     'suggestion': get_cosine_schedule_with_warmup(optimizers['suggestion'], warmup_steps, total_steps)
    # }

    # ================================= 训练 =======================================
    train_step()
