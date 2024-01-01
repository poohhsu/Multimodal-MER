import argparse
from collections import defaultdict
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from consts import emotion2id_4
from model import MyCRNN, MyShortChunkCNN_Res, CRNN, ShortChunkCNN_Res


class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_path = [p for p in os.listdir(self.data_dir) if p.split('_%%-%%_')[1] in emotion2id_4.keys()]
        self.label_mapping = emotion2id_4

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]

        return {
            'path': path,
            'data': torch.from_numpy(np.load(os.path.join(self.data_dir, path))),
            'emotion': torch.tensor(self.label_mapping[path.split('_%%-%%_')[1]]),
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
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_name == 'MyCRNN':
        model = MyCRNN(n_channels=32, n_class=len(emotion2id_4))
    elif args.model_name == 'MyShortChunkCNN_Res':
        model = MyShortChunkCNN_Res(n_channels=64, n_class=len(emotion2id_4))
    elif args.model_name == 'CRNN':
        model = CRNN(n_channels=32, n_class=len(emotion2id_4))
    elif args.model_name == 'ShortChunkCNN_Res':
        model = ShortChunkCNN_Res(n_channels=64, n_class=len(emotion2id_4))
    
    checkpoint = torch.load(os.path.join(args.ckpt_dir, f'{args.model_name}.pt'))
    model.load_state_dict(checkpoint)
    model.to(args.device)

    labels = [v['emotion'].item() for v in test_dataset]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)).to(args.device))

    total_loss, total_acc = 0, 0
    y_path, y_true, y_score = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            path, data, emotion = batch['path'], batch['data'].to(args.device), batch['emotion'].to(args.device)
            pred = model(data)
            loss = criterion(pred, emotion)

            total_loss += loss.detach().item()
            total_acc += (pred.argmax(1) == emotion).sum().item()

            y_path += path
            y_true += emotion.cpu().tolist()
            y_score += F.softmax(pred, dim=1).cpu().tolist()
    
    print('Loss:', total_loss / len(test_dataloader))
    print('Frame accuracy:', total_acc / len(y_true))

    y_pred = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title(f'{args.model_name}')
    plt.xticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.yticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.savefig('cm_cnn_frame_3.png')
    plt.clf()

    song_true, song_score = defaultdict(list), defaultdict(list)
    for path, true, score in zip(y_path, y_true, y_score):
        song_true[path.split('_%%-%%_')[0]].append(true)
        song_score[path.split('_%%-%%_')[0]].append(score)
    
    y_true, y_score = [], []
    result_dict = {}
    for k in song_true:
        y_true.append(stats.mode(song_true[k])[0])
        y_score.append(np.array(song_score[k]).mean(0))
        result_dict[k] = [y_true[-1], y_score[-1]]
    with open(os.path.join(args.save_dir, 'cnn.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)

    y_pred = np.argmax(y_score, axis=1)
    print('Song accuracy:', sum(y_pred == y_true) / len(y_true))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title(f'{args.model_name}')
    plt.xticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.yticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.savefig('cm_cnn_song_3.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='mel_5/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_cnn/')
    parser.add_argument('--save_dir', type=str, default='result_dict/')
    parser.add_argument('--model_name', type=str, default='CRNN')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    main(args)

