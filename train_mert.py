import argparse
import os
import random

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from consts import emotion2id_4
from model import MyMERT


class EmotionDataset(Dataset):
    def __init__(self, data_dir, resampler, processor):
        self.data_dir = data_dir
        self.data_path = [p for p in os.listdir(self.data_dir) if p.split('_%%-%%_')[1] in emotion2id_4.keys()]
        self.label_mapping = emotion2id_4
        self.resampler = resampler
        self.processor = processor

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        data = torch.from_numpy(np.load(os.path.join(self.data_dir, path)))
        if self.resampler is not None:
            data = self.resampler(data)

        return {
            'data': self.processor(data, sampling_rate=self.processor.sampling_rate, return_tensors='pt'),
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

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    resample_rate = processor.sampling_rate
    resampler = None
    if resample_rate != args.sr:
        resampler = T.Resample(args.sr, resample_rate)

    train_dataset = EmotionDataset(os.path.join(args.data_dir, 'train'), resampler, processor)
    valid_dataset = EmotionDataset(os.path.join(args.data_dir, 'valid'), resampler, processor)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MyMERT(args.model_name, n_class=len(emotion2id_4))
    model.to(args.device)

    labels = [v['emotion'].item() for v in train_dataset]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)).to(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=1/60)
    
    max_acc, count = 0, 0
    for epoch in range(args.num_epoch):
        print(f'==================== Epoch {epoch + 1}/{args.num_epoch} ====================')

        total_loss, total_acc = 0, 0
        pbar = tqdm(total=len(train_dataloader), ncols=0, desc='Train', unit=' step', position=0)
        model.train()
        for i, batch in enumerate(train_dataloader):
            data, emotion = batch['data'].to(args.device), batch['emotion'].to(args.device)
            pred = model((epoch < 2), **data)
            loss = criterion(pred, emotion)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total_loss += loss.detach().item()
            total_acc += (pred.argmax(1) == emotion).sum().item()

            pbar.update()
            pbar.set_postfix(
                lr=scheduler.get_last_lr()[0],
                loss=total_loss / (i + 1),
                acc=total_acc / (i + 1) / args.batch_size,
            )
        pbar.close()
        
        total_loss, total_acc = 0, 0
        pbar = tqdm(total=len(valid_dataloader), ncols=0, desc='Valid', unit=' step', position=0)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                data, emotion = batch['data'].to(args.device), batch['emotion'].to(args.device)
                pred = model(**data)
                loss = criterion(pred, emotion)

                total_loss += loss.detach().item()
                total_acc += (pred.argmax(1) == emotion).sum().item()

                pbar.update()
                pbar.set_postfix(
                    loss=total_loss / (i + 1),
                    acc=total_acc / (i + 1) / args.batch_size,
                )
        pbar.close()
        
        if total_acc / len(valid_dataloader.dataset) > max_acc:
            count = 0
            max_acc = total_acc / len(valid_dataloader.dataset)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'MyMERT.pt'))
        else:
            count += 1
            if count >= args.early_stop:
                break

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='seg_5/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_mert/')
    parser.add_argument('--model_name', type=str, default='m-a-p/MERT-v1-95M')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    main(args)

