import argparse
import os
import random

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
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

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train_dataset = EmotionDataset(os.path.join(args.data_dir, 'train'))
    valid_dataset = EmotionDataset(os.path.join(args.data_dir, 'valid'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model_name == 'MyCRNN':
        model = MyCRNN(n_channels=32, n_class=len(emotion2id_4)).to(args.device)
    elif args.model_name == 'MyShortChunkCNN_Res':
        model = MyShortChunkCNN_Res(n_channels=64, n_class=len(emotion2id_4)).to(args.device)
    elif args.model_name == 'CRNN':
        model = CRNN(n_channels=32, n_class=len(emotion2id_4)).to(args.device)
    elif args.model_name == 'ShortChunkCNN_Res':
        model = ShortChunkCNN_Res(n_channels=64, n_class=len(emotion2id_4)).to(args.device)

    labels = [v['emotion'].item() for v in train_dataset]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)).to(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch)
    
    max_acc, count = 0, 0
    for epoch in range(args.num_epoch):
        print('==================== Epoch {:2d}/{} ===================='.format(epoch + 1, args.num_epoch))

        total_loss, total_acc = 0, 0
        pbar = tqdm(total=len(train_dataloader), ncols=0, desc='Train', unit=' step', position=0)
        model.train()
        for i, batch in enumerate(train_dataloader):
            data, emotion = batch['data'].to(args.device), batch['emotion'].to(args.device)
            pred = model(data)
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
                pred = model(data)
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
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'{args.model_name}.pt'))
        else:
            count += 1
            if count >= args.early_stop:
                break

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='mel_5/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_cnn/')
    parser.add_argument('--model_name', type=str, default='CRNN')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--rand_seed', type=int, default=2023)

    args = parser.parse_args()

    main(args)

