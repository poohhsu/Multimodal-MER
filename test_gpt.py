import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from consts import emotion2id_4


def main(args):
    count, count2, total_acc = 0, 0, 0
    y_true, y_pred = [], []
    for path in tqdm(os.listdir(args.data_dir)):
        emotion = os.path.splitext(path)[0].split('_%%-%%_')[-1]
        with open(os.path.join(args.data_dir, path)) as f:
            data = f.read()
        
        pred = data
        if data not in emotion2id_4.keys():
            count += 1
            for e in emotion2id_4.keys():
                if e in data:
                    pred = e
                    break
            
            if pred == data:
                count2 += 1
                print(path)
                print(data)
                print()
                continue
        
        y_true.append(emotion2id_4[emotion])
        y_pred.append(emotion2id_4[pred])
        total_acc += (emotion == pred)
        
    print(len(y_true), count, count2)
    print('Accuracy:', total_acc / len(y_true))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title(f'Accuracy: {total_acc / len(y_true):.4f}')
    plt.xticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.yticks(np.arange(len(emotion2id_4)), emotion2id_4.keys())
    plt.savefig('cm_gpt.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='gpt/test/')

    args = parser.parse_args()

    main(args)