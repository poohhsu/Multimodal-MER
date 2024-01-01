import argparse
import os


def write_txt(data_path, write_list):
    with open(data_path, 'w', encoding='utf-8') as f:
        f.writelines('')
    with open(data_path, 'a', encoding='utf-8') as f:
        for L in write_list:
            f.writelines(f'{L[0]}, {L[1]}\n')


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train, valid, test = [], [], []
    for root, dirs, files in os.walk(args.data_dir):
        data = []
        count = 0
        for file in sorted(files):
            if os.path.splitext(file)[1] in ['.wav', '.mp3']:
                data.append([os.path.join(root, file), os.path.split(root)[-1]])
                count += 1
        print(root, count)
        
        l = len(data)
        train += data[:round(l*args.train_ratio)]
        valid += data[round(l*args.train_ratio):round(l*(args.train_ratio+args.valid_ratio))]
        test += data[round(l*(args.train_ratio+args.valid_ratio)):]

    write_txt(os.path.join(args.save_dir, 'train.txt'), train)
    write_txt(os.path.join(args.save_dir, 'valid.txt'), valid)
    write_txt(os.path.join(args.save_dir, 'test.txt'), test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='MER31k/')
    parser.add_argument('--save_dir', type=str, default='data_path/')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)

    args = parser.parse_args()

    main(args)
