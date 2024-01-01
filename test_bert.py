import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from consts import emotion2id_4


class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_path = [p for p in os.listdir(self.data_dir) if os.path.splitext(p)[0].split('_%%-%%_')[-1] in emotion2id_4.keys()]
        self.label_mapping = emotion2id_4

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        with open(os.path.join(self.data_dir, path)) as f:
            data = f.read()

        return {
            'path': path,
            'data': data,
            'emotion': torch.tensor(self.label_mapping[os.path.splitext(path)[0].split('_%%-%%_')[-1]]),
        }


def main(args):
    random.seed(args.rand_seed)  
    np.random.seed(args.rand_seed)  
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
        torch.cuda.manual_seed_all(args.rand_seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_dataset = EmotionDataset(os.path.join(args.data_dir, 'test'))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(emotion2id_4)).to(args.device)

    labels = [v['emotion'].item() for v in test_dataset]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)).to(args.device))
    
    total_loss, total_acc = 0, 0
    y_path, y_true, y_score = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            path, data, emotion = batch['path'], batch['data'], batch['emotion'].to(args.device)
            data = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
            pred = model(**data, labels=emotion)
            loss = criterion(pred.logits, emotion)

            total_loss += loss.detach().item()
            total_acc += (pred.logits.argmax(1) == emotion).sum().item()
            
            y_path += path
            y_true += emotion.cpu().tolist()
            y_score += F.softmax(pred.logits, dim=1).cpu().tolist()

    print('Loss:', total_loss / len(test_dataloader))
    print('Accuracy:', total_acc / len(y_true))

    result_dict = {}
    for path, true, score in zip(y_path, y_true, y_score):
        result_dict[path.split('_%%-%%_')[0]] = [true, score]
    with open(os.path.join(args.save_dir, 'bert.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)

    y_pred = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title('BERT')
    plt.xticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.yticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.savefig('cm_bert.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='lyrics_htdemucs/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_bert/')
    parser.add_argument('--save_dir', type=str, default='result_dict/')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args)

