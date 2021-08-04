import os 
import shutil
import numpy as np 
import random


def split_train_test_val(datapath, delete = False):
    lengths = {}
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    splits = ['test', 'val', 'train']
    for i in classes:
        lengths[i] = len(os.listdir(os.path.join(datapath, i)))

    if 'test' not in os.listdir(datapath) and 'train' not in os.listdir(datapath) and 'val' not in os.listdir(datapath):
        os.makedirs(os.path.join(datapath, 'test'))
        os.makedirs(os.path.join(datapath, 'train'))
        os.makedirs(os.path.join(datapath, 'val'))

        for i in splits:
            if len(os.listdir(os.path.join(datapath, i))) == 0:
                for class_ in classes:
                    os.makedirs(os.path.join(datapath, i, class_))


        for length in lengths:
            imgs = os.listdir(os.path.join(datapath, length))
            random.seed(42)
            random.shuffle(imgs)
            first_cutoff = int(np.floor(0.2 * lengths[length]))
            second_cutoff = first_cutoff + int(np.floor(0.2 * lengths[length]))

            test = imgs[: first_cutoff]
            val = imgs[first_cutoff : second_cutoff]
            train = imgs[second_cutoff : ]

            for i in test:
                shutil.copy(os.path.join(datapath, length, i), os.path.join(datapath, 'test', length, i))
            for i in val:
                shutil.copy(os.path.join(datapath, length, i), os.path.join(datapath, 'val', length, i))
            for i in train:
                shutil.copy(os.path.join(datapath, length, i), os.path.join(datapath, 'train', length, i))


            if delete == True:
                shutil.rmtree(os.path.join(datapath, length))

    return



if __name__ == '__main__':
    split_train_test_val(r'C:\Users\emirdemic\Desktop\CNN-Alzheimer-Predictions\Not split')