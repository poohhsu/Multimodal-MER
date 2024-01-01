import argparse
import os

import demucs.separate


def main(args):
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if os.path.splitext(file)[1] in ['.wav', '.mp3']:
                song_path = os.path.join(root, file)
                if not os.path.exists(os.path.join('htdemucs', song_path)):
                    demucs.separate.main(['-d', args.device, '--mp3-preset', args.preset, '--two-stems', 'vocals', '--other-method', 'none', '-o', args.save_dir, '--filename', song_path, song_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='MER31k/')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--preset', type=str, default='2')

    args = parser.parse_args()

    main(args)