# coding: utf-8

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import MakeDataset
from tqdm import tqdm
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore") # 忽略warning


# 参数设置
model_path = r'./models/bert-base-chinese/'
data_select = 'med'
data_path = r'./data/' + data_select + '/'
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 3
learning_rate = 4e-6  # Learning Rate不宜太大

def train():
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 获取到dataset
    train_dataset = MakeDataset(data_path + data_select + '.train.txt', tokenizer)
    valid_dataset = MakeDataset(data_path + data_select + '.val.txt', tokenizer)
    # test_dataset = MakeDataset(data_path + data_select+ '.test.txt', tokenizer)

    # 生成Batch
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # 验证
        model.eval()
        losses = 0  # 损失
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label)  # acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        print('\tLoss:', average_loss)

        # 分类报告
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id,
                                               target_names=valid_dataset.labels)
        print('* Classification Report:')
        print(report)

        # f1 用来判断最优模型
        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')

        if not os.path.exists('models/'+data_select+'/'):
            os.makedirs('models/'+data_select+'/')

        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'models/' + data_select + '/best_model.pkl')


def test():
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 获取到dataset
    test_dataset = MakeDataset(data_path + data_select + '.test.txt', tokenizer)

    # 生成Batch
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(test_dataset.labels)

    # 加载模型
    model = BertClassifier(bert_config, num_labels).to(device)
    model.load_state_dict(torch.load('./models/' + data_select + '/best_model.pkl', map_location=torch.device('cpu')))

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 测试
    model.eval()
    losses = 0  # 损失
    pred_labels = []
    true_labels = []
    test_bar = tqdm(test_dataloader, ncols=100)
    for input_ids, token_type_ids, attention_mask, label_id in test_bar:
        test_bar.set_description('Test')

        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
        )

        loss = criterion(output, label_id.to(device))
        losses += loss.item()

        pred_label = torch.argmax(output, dim=1)  # 预测出的label
        acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label)  # acc
        test_bar.set_postfix(loss=loss.item(), acc=acc)

        pred_labels.extend(pred_label.cpu().numpy().tolist())
        true_labels.extend(label_id.numpy().tolist())

    average_loss = losses / len(test_dataloader)
    print('\tLoss:', average_loss)
    # 分类报告
    report = metrics.classification_report(true_labels, pred_labels, labels=test_dataset.labels_id,
                                           target_names=test_dataset.labels)
    print('* Classification Report:')
    print(report)


if __name__ == '__main__':
    train()
    # test()
    print("Done")