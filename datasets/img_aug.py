# source: https://github.com/cfchen-duke/ProtoPNet/blob/master/img_aug.py

import Augmentor
import os
from utils.helpers import makedir


def image_augment(datasets_root_dir, target_root_dir, source='train', target='train_augmented'):

    dir = os.path.join(datasets_root_dir, source)
    target_dir = os.path.join(target_root_dir, target)

    makedir(target_dir)
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

    for i in range(len(folders)):
        fd = folders[i]
        tfd = target_folders[i]
        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # random_distortion
        # p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        # p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        # p.flip_left_right(probability=0.5)
        # for i in range(10):
        #    p.process()
        # del p



if __name__ == '__main__':
    # datasets_root_dir = '/home/bizon/Workspace/ProtoPNet/ProtoPNet-master/datasets_cub_cropped/cub200_cropped/'
    datasets_root_dir = '/home/bizon/Downloads/'
    # dir = datasets_root_dir + 'train/'
    dir = datasets_root_dir + 'car_train/'
    # target_dir = datasets_root_dir + 'train_augmented/'
    target_dir = dir + 'train_augmented/'
    image_augment(datasets_root_dir, target_dir, source='train', target='')