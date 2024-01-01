import argparse
import os

from openai import OpenAI
from tqdm import tqdm
import whisper


def get_lyrics(src, args):
    if not os.path.exists(os.path.join(args.save_dir, src)):
        os.makedirs(os.path.join(args.save_dir, src))

    model = whisper.load_model('large-v3')
    with open(os.path.join(args.data_dir, f'{src}.txt'), encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            path, emotion = line.strip().rsplit(', ', 1)
            if os.path.getsize(os.path.join(args.save_dir, src, f'{os.path.splitext(os.path.split(path)[-1])[0]}_%%-%%_{emotion}.txt')) < 64:
                transcript = model.transcribe(path)

                with open(os.path.join(args.save_dir, src, f'{os.path.splitext(os.path.split(path)[-1])[0]}_%%-%%_{emotion}.txt'), 'w') as f2:
                    for segment in transcript['segments']:
                        f2.write(f"{segment['text'].strip()}\n")


def main(args):
    get_lyrics('train', args)
    get_lyrics('valid', args)
    get_lyrics('test', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data_path_htdemucs/')
    parser.add_argument('--save_dir', type=str, default='lyrics_htdemucs/')

    args = parser.parse_args()

    main(args)