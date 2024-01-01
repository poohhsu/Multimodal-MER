import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from consts import emotion2id_4


def main(args):
    result = []
    with open(os.path.join(args.data_dir, 'cnn.pkl'), 'rb') as f:
        result_cnn = pickle.load(f)
    with open(os.path.join(args.data_dir, 'mert.pkl'), 'rb') as f:
        result_mert = pickle.load(f)
    with open(os.path.join(args.data_dir, 'bert.pkl'), 'rb') as f:
        result_bert = pickle.load(f)
    # with open(os.path.join(args.data_dir, 'remi.pkl'), 'rb') as f:
    #     result_remi = pickle.load(f)

    y_true, y_score = [], []
    for k in result_cnn:
        y_true.append(stats.mode([result_cnn[k][0], result_mert[k][0], result_bert[k][0]])[0])
        y_score.append([a + b + c for a, b, c in zip(result_cnn[k][1], result_mert[k][1], result_bert[k][1])])
        # y_true.append(stats.mode([result_cnn[k][0], result_mert[k][0], result_bert[k][0], result_remi[k][0]])[0])
        # y_score.append([a + b + c + d for a, b, c, d in zip(result_cnn[k][1], result_mert[k][1], result_bert[k][1], result_remi[k][1])])

    y_pred = np.argmax(y_score, axis=1)
    print('Accuracy:', sum(y_pred == y_true) / len(y_true))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title(f'Accuracy: {sum(y_pred == y_true) / len(y_true):.4f}')
    plt.xticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.yticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.savefig('cm_late_fusion.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='result_dict/')

    args = parser.parse_args()

    main(args)
