import argparse
import os

import librosa
import numpy as np
from tqdm import tqdm


def get_seg(src, args):
    if not os.path.exists(os.path.join(args.save_dir, src)):
        os.makedirs(os.path.join(args.save_dir, src))

    sample_len = args.seg_len * args.sr
    with open(os.path.join(args.data_dir, f'{src}.txt'), encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            path, emotion = line.strip().rsplit(', ', 1)
            y, sr = librosa.load(path, sr=args.sr)
            count = 0
            for start in range(0, y.shape[0], sample_len):
                if start + sample_len <= y.shape[0]:
                    seg = y[start:start+sample_len]
                    if np.sqrt(np.mean(np.abs(seg) ** 2)) > args.threshold: 
                        np.save(os.path.join(args.save_dir, src, f'{os.path.splitext(os.path.split(path)[-1])[0]}_%%-%%_{emotion}_%%-%%_{count}'), seg)
                        count += 1


def main(args):
    get_seg('train', args)
    get_seg('valid', args)
    get_seg('test', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data_path/')
    parser.add_argument('--save_dir', type=str, default='seg_5/')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--seg_len', type=int, default=5)
    parser.add_argument('--threshold', type=int, default=0.01)

    args = parser.parse_args()

    main(args)
