import urllib.request
import urllib.error
import glob
import os
import functools
import re
from pathlib import Path
import random
from multiprocessing.dummy import Pool  # use threads for I/O bound tasks


def download_img(url, dir):
    try:
        filename = os.path.join(dir, re.match(r".*([A-Za-z0-9_]{32}.jpg)", url).group(1))
        my_file = Path(filename)
        if my_file.is_file():
            pass
        else:
            urllib.request.urlretrieve(url.rstrip(), filename)
    except urllib.error.HTTPError as e:
        print("HTTP Error 404 %s", url.rstrip())


if __name__ == "__main__":
    # files = glob.glob(os.path.join(os.getcwd(), '_data/img_url/*.txt'))
    # print(files)
    #
    # for file in files:
    #     c = re.match(r".*img_url/[0-9]-(.*).txt", file).group(1)
    #     os.mkdir(os.path.join(os.getcwd(), '_data', c))
    #     directory = os.path.join(os.getcwd(), '_data', c)
    #     # open the file and then call .read() to get the text
    #     with open(file) as f:
    #         urls = f.readlines()
    #         Pool(processes=8).map(functools.partial(download_img, dir=directory), urls)

    # classes = ['env', 'food', 'front', 'menu', 'profile']
    #
    # for c in classes:
    #     os.mkdir(os.path.join(os.getcwd(), '_data', 'train', c))
    #     os.mkdir(os.path.join(os.getcwd(), '_data', 'validate', c))
    #     directory = os.path.join(os.getcwd(), '_data', c, '*.jpg')
    #     files = glob.glob(directory)
    #     random.shuffle(files)
    #
    #     train = files[:int(len(files) * 0.8)]
    #     validate = files[int(len(files) * 0.8):]
    #     assert (set(train).intersection(validate) == set())
    #
    #     for f in train:
    #         new_dir = re.sub('_data/', '_data/train/', f)
    #         os.rename(f, new_dir)
    #     for f in validate:
    #         new_dir = re.sub('_data/', '_data/validate/', f)
    #         os.rename(f, new_dir)

    classes = ['env', 'food', 'front', 'menu', 'profile']

    for c in classes:
        os.makedirs(os.path.join(os.getcwd(), '_data/validate', 'validate', c))
        os.makedirs(os.path.join(os.getcwd(), '_data/validate', 'test', c))
        directory = os.path.join(os.getcwd(), '_data/validate', c, '*.jpg')
        files = glob.glob(directory)
        random.shuffle(files)

        train = files[:int(len(files) * 0.75)]
        test = files[int(len(files) * 0.75):]
        assert (set(train).intersection(test) == set())

        for f in train:
            new_dir = re.sub('_data/validate/', '_data/validate/validate/', f)
            os.rename(f, new_dir)
        for f in test:
            new_dir = re.sub('_data/validate/', '_data/validate/test/', f)
            os.rename(f, new_dir)


