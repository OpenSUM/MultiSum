import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None)
parser.add_argument('--dst', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)


def word2char(sentence, remove_tags=False, add_last_tag=False, keep_least_one_tag_for_rouge=False):
    words = sentence.split(' ')
    chars = []
    for w in words:
        if w in ('<s>', '</s>', '<unk>', '<pad>'):
            if not remove_tags:
                chars.append(w)
        elif w in ['<ln>']:
            chars.append(w)
        else:
            s = sum([1 if '\u4e00' <= c <= '\u9fff' else 0 for c in w])
            if s > 0:
                for c in w:
                    chars.append(c)
            else:
                chars.append(w)
    if add_last_tag:
        chars.append('</s>')
    if keep_least_one_tag_for_rouge and len(chars) == 0:
        chars.append('</s>')
    return ' '.join(chars)


def word2char_file(src_file, dst_file):
    with open(src_file, 'r', encoding='utf8') as fsrc, open(dst_file, 'w', encoding='utf8') as fdst:
        for line in fsrc:
            fdst.write(word2char(line.strip(), remove_tags=True, keep_least_one_tag_for_rouge=True))
            fdst.write('\n')


def data_transform():
    for filename in ['test.src', 'test.dst', 'eval.src', 'eval.dst', 'train.src', 'train.dst']:
        filepath = os.path.join('./word', filename + '.token')
        with open(filepath, 'r', encoding='utf8') as f:
            lines = [word2char(l.strip()) for l in f]
        filepath = os.path.join('./char', filename + '.token')
        with open(filepath, 'w', encoding='utf8') as f:
            for l in lines:
                f.write(l)
                f.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.ckpt:
        print('Converting files in', args.ckpt)
        source_dir = args.ckpt
        target_dir = os.path.join(source_dir, 'split')
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(source_dir):
            filepath = os.path.join(source_dir, filename)
            if os.path.isfile(filepath) and not filepath.endswith('.log'):
                print('Converting file:', filepath)
                word2char_file(filepath, os.path.join(target_dir, filename))
    else:
        word2char_file(args.src, args.dst)
