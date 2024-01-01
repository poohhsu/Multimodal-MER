import argparse
import os
import time

from openai import OpenAI
from tqdm import tqdm

from consts import emotion2id_4


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == 0: # for binary classification
        prompt = 'Identify the emotions conveyed in the lyrics: Angry, Anxious, Bright, Happy, Lazy, Messy, Peaceful, Sad. Choose the emotions that are present.'
    elif args.mode == 1: # for multiclass classification
        prompt = 'Identify the emotion conveyed in the lyrics: Angry, Happy, Peaceful, Sad. Choose one.'

    client = OpenAI(api_key='your-openai-api-key')
    for path in tqdm(os.listdir(args.data_dir)):
        emotion = os.path.splitext(path)[0].split('_%%-%%_')[-1]
        if args.mode == 0 or emotion in emotion2id_4.keys():
            with open(os.path.join(args.data_dir, path)) as f:
                data = f.read()

            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'system',
                        'content': prompt
                    },
                    {
                        'role': 'user',
                        'content': f'Lyrics: {data}'
                    }
                ]
            )

            with open(os.path.join(args.save_dir, path), 'w') as f:
                f.write(response.choices[0].message.content)
            
            time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='lyrics_htdemucs/test/')
    parser.add_argument('--save_dir', type=str, default='gpt/test/')
    parser.add_argument('--mode', type=int, default=0, help='0: binary, 1: multiclass')

    args = parser.parse_args()

    main(args)