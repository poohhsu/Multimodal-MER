import argparse
from collections import defaultdict
import os
import pickle
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from consts import emotion2id
from model import MyMERT


class EmotionDataset(Dataset):
    def __init__(self, data_dir, resampler, processor, emotion_id):
        self.data_dir = data_dir
        self.data_path = os.listdir(self.data_dir)
        self.label_mapping = emotion2id
        self.resampler = resampler
        self.processor = processor
        self.emotion_id = emotion_id

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        data = torch.from_numpy(np.load(os.path.join(self.data_dir, path)))
        if self.resampler is not None:
            data = self.resampler(data)

        return {
            'path': path,
            'data': self.processor(data, sampling_rate=self.processor.sampling_rate, return_tensors='pt'),
            'emotion': torch.tensor(int(self.label_mapping[path.split('_%%-%%_')[1]] == self.emotion_id)),
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

    result_frame, result_song = [], []
    for emotion_id in range(len(emotion2id)):
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
        resample_rate = processor.sampling_rate
        resampler = None
        if resample_rate != args.sr:
            resampler = T.Resample(args.sr, resample_rate)

        test_dataset = EmotionDataset(os.path.join(args.data_dir, 'test'), resampler, processor, emotion_id)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = MyMERT(args.model_name)
        checkpoint = torch.load(os.path.join(args.ckpt_dir, f'MyMERT_{emotion_id}.pt'))
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
                pred = model(**data)
                loss = criterion(pred, emotion)

                total_loss += loss.detach().item()

                y_path += path
                y_true += emotion.cpu().tolist()
                y_score += pred.softmax(1).cpu().tolist()
        
        y_pred = np.argmax(y_score, axis=1)
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, np.array(y_score)[:, 1])

        result_frame.append([
            total_loss / len(test_dataloader),
            total_acc / len(test_dataloader.dataset),
            P,
            R,
            F,
            auc
        ])

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
        with open(os.path.join(args.save_dir, f'mert_{emotion_id}.pkl'), 'wb') as f:
            pickle.dump(result_dict, f)

        y_pred = np.argmax(y_score, axis=1)
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, np.array(y_score)[:, 1])

        result_song.append([
            sum(y_pred == y_true) / len(y_true),
            P,
            R,
            F,
            auc
        ])

    result_frame.append(np.mean(result_frame, axis=0))
    pd.DataFrame(result_frame, index=list(emotion2id.keys())+['Average']).to_csv('mert_frame.csv', header=[
        'Loss',
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'AUC'
    ])

    result_song.append(np.mean(result_song, axis=0))
    pd.DataFrame(result_song, index=list(emotion2id.keys())+['Average']).to_csv('mert_song.csv', header=[
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'AUC'
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='seg_5/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_mert_binary/')
    parser.add_argument('--save_dir', type=str, default='result_dict_binary/')
    parser.add_argument('--model_name', type=str, default='m-a-p/MERT-v1-95M')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    main(args)

