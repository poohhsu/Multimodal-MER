import argparse
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm


def get_feature(src, args):
    if not os.path.exists(os.path.join(args.mel_dir, src)):
        os.makedirs(os.path.join(args.mel_dir, src))
    if not os.path.exists(os.path.join(args.f0_dir, src)):
        os.makedirs(os.path.join(args.f0_dir, src))

    for path in tqdm(os.listdir(os.path.join(args.data_dir, src))):
        if not os.path.exists(os.path.join(args.f0_dir, src, path)):
            y = np.load(os.path.join(args.data_dir, src, path))
            S = librosa.feature.melspectrogram(y=y, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
            log_S = librosa.power_to_db(S, ref=np.max)
            np.save(os.path.join(args.mel_dir, src, path[:-4]), log_S)
            log_S = np.load(os.path.join(args.mel_dir, src, path))


def main(args):
    get_feature('train', args)
    get_feature('valid', args)
    get_feature('test', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='seg_5/')
    parser.add_argument('--mel_dir', type=str, default='mel_5/')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--device', type=torch.device, default='cuda')

    args = parser.parse_args()

    main(args)
