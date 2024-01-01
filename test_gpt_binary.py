import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from consts import emotion2id


def main(args):
    id2emotion = {v: k for k, v in emotion2id.items()}
    result = []
    for emotion_id in range(len(emotion2id)):
        y_true, y_pred = [], []
        for path in tqdm(os.listdir(args.data_dir)):
            emotion = os.path.splitext(path)[0].split('_%%-%%_')[-1]
            with open(os.path.join(args.data_dir, path)) as f:
                data = f.read()

            y_true.append(int(emotion2id[emotion] == emotion_id))
            y_pred.append(int(id2emotion[emotion_id] in data))
            
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_pred)

        result.append([
            sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true),
            P,
            R,
            F,
            auc
        ])

    result.append(np.mean(result, axis=0))
    pd.DataFrame(result, index=list(emotion2id.keys())+['Average']).to_csv('gpt.csv', header=[
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'AUC'
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='gpt_binary/test/')

    args = parser.parse_args()

    main(args)