import argparse
import os
import random

import numpy as np
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

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for emotion_id in range(len(emotion2id)):
        print(f'\n================================ Emotion {emotion_id} ================================\n')
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
        resample_rate = processor.sampling_rate
        resampler = None
        if resample_rate != args.sr:
            resampler = T.Resample(args.sr, resample_rate)

        train_dataset = EmotionDataset(os.path.join(args.data_dir, 'train'), resampler, processor, emotion_id)
        valid_dataset = EmotionDataset(os.path.join(args.data_dir, 'valid'), resampler, processor, emotion_id)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = MyMERT(args.model_name)
        model.to(args.device)

        labels = [v['emotion'].item() for v in train_dataset]
        criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)).to(args.device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=1/60)
        
        max_auc, count = 0, 0
        for epoch in range(args.num_epoch):
            print(f'==================== Epoch {epoch + 1}/{args.num_epoch} ====================')

            total_loss, total_acc = 0, 0
            y_true, y_score, y_pred = [], [], []
            pbar = tqdm(total=len(train_dataloader), ncols=0, desc='Train', unit=' step', position=0)
            model.train()
            for i, batch in enumerate(train_dataloader):
                data, emotion = batch['data'].to(args.device), batch['emotion'].to(args.device)
                pred = model((epoch < 6), **data)
                loss = criterion(pred, emotion)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                total_loss += loss.detach().item()
                total_acc += (pred.argmax(1) == emotion).sum().item()

                y_true += emotion.cpu().tolist()
                y_score += pred.softmax(1).cpu().tolist()
                y_pred += pred.argmax(1).cpu().tolist()

                P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                auc = roc_auc_score(y_true, np.array(y_score)[:, 1]) if i > 10 else 0

                pbar.update()
                pbar.set_postfix(
                    lr=scheduler.get_last_lr()[0],
                    loss=total_loss / (i + 1),
                    acc=total_acc / (i + 1) / args.batch_size,
                    P=P,
                    R=R,
                    F=F,
                    auc=auc,
                )
            pbar.close()
            
            total_loss, total_acc = 0, 0
            y_true, y_score, y_pred = [], [], []
            pbar = tqdm(total=len(valid_dataloader), ncols=0, desc='Valid', unit=' step', position=0)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(valid_dataloader):
                    data, emotion = batch['data'].to(args.device), batch['emotion'].to(args.device)
                    pred = model(**data)
                    loss = criterion(pred, emotion)

                    total_loss += loss.detach().item()
                    total_acc += (pred.argmax(1) == emotion).sum().item()

                    y_true += emotion.cpu().tolist()
                    y_score += pred.softmax(1).cpu().tolist()
                    y_pred += pred.argmax(1).cpu().tolist()

                    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                    auc = roc_auc_score(y_true, np.array(y_score)[:, 1]) if i > 10 else 0

                    pbar.update()
                    pbar.set_postfix(
                        loss=total_loss / (i + 1),
                        acc=total_acc / (i + 1) / args.batch_size,
                        P=P,
                        R=R,
                        F=F,
                        auc=auc,
                    )
            pbar.close()

            if auc > max_auc:
                count = 0
                max_auc = auc
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'MyMERT_{emotion_id}.pt'))
            else:
                count += 1
                if count >= args.early_stop:
                    break

            scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='seg_5/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_mert_binary/')
    parser.add_argument('--model_name', type=str, default='m-a-p/MERT-v1-95M')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    main(args)

