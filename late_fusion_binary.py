import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from consts import emotion2id


def main(args):
    result = []
    for emotion_id in range(len(emotion2id)):
        with open(os.path.join(args.data_dir, f'cnn_{emotion_id}.pkl'), 'rb') as f:
            result_cnn = pickle.load(f)
        with open(os.path.join(args.data_dir, f'mert_{emotion_id}.pkl'), 'rb') as f:
            result_mert = pickle.load(f)
        with open(os.path.join(args.data_dir, f'bert_{emotion_id}.pkl'), 'rb') as f:
            result_bert = pickle.load(f)
        # with open(os.path.join(args.data_dir, f'remi_{emotion_id}.pkl'), 'rb') as f:
        #     result_remi = pickle.load(f)

        y_true, y_score = [], []
        for k in result_cnn:
            y_true.append(stats.mode([result_cnn[k][0], result_mert[k][0], result_bert[k][0]])[0])
            y_score.append([a + b + c for a, b, c in zip(result_cnn[k][1], result_mert[k][1], result_bert[k][1])])
            # y_true.append(stats.mode([result_cnn[k][0], result_mert[k][0], result_bert[k][0], result_remi[k][0]])[0])
            # y_score.append([a + b + c + d for a, b, c, d in zip(result_cnn[k][1], result_mert[k][1], result_bert[k][1], result_remi[k][1])])

        y_pred = np.argmax(y_score, axis=1)
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, np.array(y_score)[:, 1])

        result.append([
            sum(y_pred == y_true) / len(y_true),
            P,
            R,
            F,
            auc
        ])

    result.append(np.mean(result, axis=0))
    pd.DataFrame(result, index=list(emotion2id.keys())+['Average']).to_csv('late_fusion.csv', header=[
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'AUC'
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='result_dict_binary/')

    args = parser.parse_args()

    main(args)
