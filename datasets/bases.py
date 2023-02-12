import os
import random
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
# from preprocess import mean, std  # todo: use . when call main.py, to be fixed
from .preprocess import mean, std
# from img_aug import image_augment  # todo: use . when call main.py, to be fixed
from .img_aug import image_augment


def get_annot_info(annot):
    annot = annot.split('\n')[0]
    sid, label = annot.split()
    sid = sid.split('/')[-1].split('.')[0]
    label = int(label)
    return sid, label


def get_annot_aug_info(annot):
    annot = annot.split('\n')[0]
    sid, label = annot.split()
    sid = sid.split('/')[-1].split('.')[0]
    sid = sid.split('_')[-1]
    label = int(label)
    return sid, label


def _read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def show_image(file):
    """
    Arg:
        - file: (.jpg, .png, ...)
    """
    img = read_image(file)
    transform = transforms.ToPILImage()
    img_ = transform(img)
    img_.show()


def get_data(line):
    img, label = line.split()[0], line.split()[1]
    return img, label


class BaseDataset(object):
    """
    Base class of image classification dataset
    Transform folder structure to .txt file

    Example:
        Output
            Art.txt
                Art/Alarm_Clock/00001.jpg 0
                Art/Alarm_Clock/00002.jpg 0
                ...
            Art_augmented.txt
                Art_augmented/Alarm_Clock/Alarm_Clock_original_00004.jpg_4e2162cc-9c07-4e59-91cc-61c0d4e51ad9.jpg 0
                Art_augmented/Alarm_Clock/Alarm_Clock_original_00004.jpg_54bc1b7d-230d-4c46-84b3-7f91f57de5e5.jpg 0
                ...
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_root_dir = os.path.dirname(self.dataset_dir)
        self.num_class_dict = None
        self.class_num_dict = None
        self._class_num_mapping()
        self.classes = list(self.class_num_dict)

    def _class_num_mapping(self):
        classes = sorted(os.listdir(self.dataset_dir))  # assume each domains have the same classes
        class_to_num = {}
        num_to_class = {}
        for i, cls in enumerate(classes):
            class_to_num[cls] = i
            num_to_class[i] = cls

        self.num_class_dict = num_to_class
        self.class_num_dict = class_to_num

    def print_data_statistics(self):
        print(' ==== Classes ====')
        for num, cls in self.num_class_dict.items():
            print(num + 1, cls)

    def augment(self, target_root_dir, suffix='_augmented'):
        """
        augment images in <Base> folder into <target_root_dir>/<Base_augmented> folder.

        Example:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_org'
            augment_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_aug'

            for domain in domains:
                dataset_dir = os.path.join(dataset_root_dir, domain)
                dataset = BaseDataset(dataset_dir)
                dataset.augment(augment_root_dir)

        """
        basename = os.path.basename(self.dataset_dir)
        image_augment(self.dataset_dir, target_root_dir, '', basename + suffix)

    def get_annotations(self, annotation_dir=None):
        """
        Output .txt with image path and id
        default output to directory name of <self.dataset_dir>

        Example:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_org'
            augment_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_aug'

            for domain in domains:
                # original
                dataset_dir = os.path.join(dataset_root_dir, domain)
                dataset = BaseDataset(dataset_dir)
                dataset.get_annotations()

                # augmented
                dataset_dir = os.path.join(augment_root_dir, domain + '_augmented')
                dataset = BaseDataset(dataset_dir)
                dataset.get_annotations()

        Output: e.g.
             'Art.txt'
                Art/Alarm_Clock/00000.jpg 0
                Art/Alarm_Clock/00001.jpg 0
                Art/Backpack/00026.jpg 1
                Art/Backpack/00027.jpg 1
                ... ...
        """
        basename = os.path.basename(self.dataset_dir)  # use basename for .txt

        if annotation_dir is None:
            annotation_dir = self.dataset_root_dir
        annotation_file = basename + '.txt'

        annotation_file = os.path.join(annotation_dir, annotation_file)
        with open(annotation_file, 'w') as f:
            for cls in self.classes:
                files = sorted(os.listdir(self.dataset_dir + '/' + cls))
                for file in files:
                    string = basename + '/' + cls + '/' + file + ' ' + str(self.class_num_dict[cls]) + '\n'
                    f.write(string)
                    print('Writing ', string)
        print('{} created.'.format(annotation_file))


class BaseImageDataset(BaseDataset):
    """
    Tranform all annotations into train/test and train_augmented

    Example
        Input
            Art.txt
            Art_augmented.txt
        Output
            Art_train.txt
            Art_test.txt
            Art_train_augmented.txt
    """
    def __init__(self, annot, annot_augmented, train_ratio=0.8):
        super(BaseDataset).__init__()
        self.annot = annot
        self.annot_augmented = annot_augmented
        self.train_ratio = train_ratio

        self.class_sid = defaultdict(list)
        self.class_sid_train = defaultdict(list)
        self.class_sid_test = defaultdict(list)

        self.class_indices = defaultdict(list)
        self.class_indices_train = defaultdict(list)
        self.class_indices_test = defaultdict(list)

        self.class_sid_indices_augmented = defaultdict(list)
        self.class_sid_indices_augmented_train = defaultdict(list)

        self._get_class_sid()
        self._get_class_indices()
        self._get_annot_aug()

    def _get_class_sid(self):
        with open(self.annot, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                sid, label = get_annot_info(lines[i])
                self.class_sid[label].append(sid)

    def _get_class_indices(self):
        with open(self.annot, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                _, label = get_annot_info(lines[i])
                self.class_indices[label].append(i)

    def _get_annot_aug(self):
        with open(self.annot_augmented, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            sid, label = get_annot_aug_info(lines[i])
            self.class_sid_indices_augmented[(label, sid)].append(i)

    def split(self, shuffle=False):
        for label in self.class_indices.keys():
            indices = self.class_indices[label]
            if shuffle:
                random.shuffle(indices)

            num = len(indices)
            train_num = int(num * self.train_ratio)
            self.class_indices_train[label] = indices[: train_num]
            self.class_indices_test[label] = indices[train_num:]

        if shuffle:
            print('[info] train/test split: Shuffled ')
        else:
            print('[info] train/test split: No Shuffle')

    def gen_annot_split(self, out_dir=None):
        if out_dir is None:
            out_dir = os.path.dirname(self.annot)

        basename = os.path.basename(self.annot)
        basename = basename.split('.')[0]

        annot_train = os.path.join(out_dir, basename + '_train.txt')
        annot_test = os.path.join(out_dir, basename + '_test.txt')

        lines_train, lines_test = [], []

        with open(self.annot, 'r') as f:
            lines = f.readlines()
            for label in self.class_indices.keys():
                train_indices = self.class_indices_train[label]
                train_indices = sorted(train_indices)
                for idx in train_indices:
                    lines_train.append(lines[idx])

                test_indices = self.class_indices_test[label]
                test_indices = sorted(test_indices)
                for idx in test_indices:
                    lines_test.append(lines[idx])

        with open(annot_train, 'w') as f:
            for line in lines_train:
                f.write(line)

        with open(annot_test, 'w') as f:
            for line in lines_test:
                f.write(line)

        print('[info] Output {}'.format(annot_train))
        print('[info] Output {}'.format(annot_test))

    def gen_annot_split_aug(self, annot_train, mode='train', out_dir=None):
        if mode not in annot_train:
            assert 'Incorrect input, Use train data instead'

        if out_dir is None:
            out_dir = os.path.dirname(self.annot)

        basename = os.path.basename(annot_train)
        basename = basename.split('.')[0]

        annot_train_augmented = os.path.join(out_dir, basename + '_augmented' + '.txt')

        with open(annot_train, 'r') as f:
            lines = f.readlines()
            for line in lines:
                sid, label = get_annot_info(line)
                self.class_sid_indices_augmented_train[(label, sid)] = self.class_sid_indices_augmented[(label, sid)]

        with open(self.annot_augmented, 'r') as f:
            annot_augmented_lines = f.readlines()

        with open(annot_train_augmented, 'w') as f:
            for key in self.class_sid_indices_augmented_train.keys():
                indices = self.class_sid_indices_augmented_train[key]
                for idx in indices:
                    f.write(annot_augmented_lines[idx])

    def print_data_statistics(self):
        pass  # todo: to be added


class ImageDataset(Dataset):
    def __init__(self, dataset_dir, annot, transform=None, target_transform=None):
        with open(annot, 'r') as f:
            lines = f.readlines()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, label = get_data(self.lines[idx])
        img = _read_image(os.path.join(self.dataset_dir, img_name))

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, int(label)


#----------------------- RANOM PAIR SAMPLER -------------------
def gen_class_indices_dict(file):
    """
    Convert .txt into class_indices_dictionary
    Args:
        - file (.txt):
        [0] Clipart/train_augmented/Alarm_Clock/Alarm_Clock_original_00004.jpg_15172cf2-c8e4-43da-bd77-85aa6444fc1e.jpg 0
        [1] Clipart/train_augmented/Alarm_Clock/Alarm_Clock_original_00005.jpg_cb0e6dd7-a337-49b7-bcf1-46642b5e50b9.jpg 0
        [2] Clipart/train_augmented/Alarm_Clock/Alarm_Clock_original_00007.jpg_9447d2c2-a3f0-4045-bcb5-e5b341aa3df9.jpg 0
        [3] Clipart/train_augmented/Backpack/Backpack_original_00023.jpg_dd2b40ca-946b-440f-8fde-cb8bd57aefae.jpg 1
        [4] Clipart/train_augmented/Backpack/Backpack_original_00023.jpg_e0b8d821-b9b7-4c62-9b53-2f018743cf3a.jpg 1
    Return:
        - class_to_indices (dict)
            class_to_indices[0] = [0, 1, 2]
            class_to_indices[1] = [3, 4]

    """
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            _, label = get_data(lines[i])
            class_to_indices[label].append(i)
    return class_to_indices


def make_identity_random_pairs(class_to_indices_1, class_to_indices_2):
    """
    Generate Random Pairs in with SAME identitiy, Use all data
    Example
        Input:
            class_to_indices_1 = {'0': [0, 1], '1': [2, 3, 4], '2': [5, 6, 7, 8]}
            class_to_indices_2 = {'0': [0, 1, 2], '1': [3, 4, 5, 6], '2': [7, 8, 9]}
        Output;
            class_to_indices_1 = {'0': [0, 1, 0], '1': [2, 3, 4, 2], '2': [5, 6, 7, 8]}
            class_to_indices_2 = {'0': [0, 1, 2], '1': [3, 4, 5, 6], '2': [7, 8, 9, 9]}
    """
    import random
    assert class_to_indices_1.keys() == class_to_indices_2.keys()
    for k in class_to_indices_1.keys():
        len_1 = len(class_to_indices_1[k])
        len_2 = len(class_to_indices_2[k])
        """ dealing with unequal lengths """
        margin = abs(len_1 - len_2)
        if len_1 < len_2:
            class_to_indices_1[k].extend(random.choices(class_to_indices_1[k], k=margin))
        if len_1 > len_2:
            class_to_indices_2[k].extend(random.choices(class_to_indices_2[k], k=margin))
    return class_to_indices_1, class_to_indices_2


def dict_value_to_list(D):
    """
    Example
        Input:
            {'0': [0, 1, 0], '1': [2, 3, 4, 2], '2': [5, 6, 7, 8]}
        Output:
            [0, 1, 0, 2, 3, 4, 2, 5, 6, 7, 8]
    """
    from itertools import chain
    out = list(chain(*D.values()))
    return out


class CrossDomainImageDataset(Dataset):
    def __init__(self, annot_1, annot_2, dataset_dir, transform=None, target_transform=None):
        with open(annot_1, 'r') as f:
            lines_1 = f.readlines()
        with open(annot_2, 'r') as f:
            lines_2 = f.readlines()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.lines_1 = lines_1
        self.lines_2 = lines_2

        self.class_to_indices_1 = gen_class_indices_dict(annot_1)
        self.class_to_indices_2 = gen_class_indices_dict(annot_2)

    def __len__(self):
        return max(len(self.lines_1), len(self.lines_2))

    def __getitem__(self, idx):
        img_name_1, label_1 = get_data(self.lines_1[idx[0]])  # domain 1
        img_name_2, label_2 = get_data(self.lines_2[idx[1]])  # domain 2

        img_1 = _read_image(os.path.join(self.dataset_dir, img_name_1))
        img_2 = _read_image(os.path.join(self.dataset_dir, img_name_2))

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.target_transform:
            label_1 = self.target_transform(label_1)
            label_2 = self.target_transform(label_2)
        return img_1, img_2, int(label_1), int(label_2)


class RandomPairSampler(Sampler):
    '''

    '''
    def __init__(self, data_source, verbose=False):
        self.data_source = data_source
        self.verbose = verbose
        self.cls_idx_dict_1 = data_source.class_to_indices_1
        self.cls_idx_dict_2 = data_source.class_to_indices_2

        # update during iteration
        self.pairs = None
        self.num_pairs = None

    def __iter__(self):
        # update for each epoch, extend into equal lengths (use all data)
        cls_idx_ext_dict_1, cls_dix_ext_dict_2 = \
            make_identity_random_pairs(self.cls_idx_dict_1, self.cls_idx_dict_2)

        # 1. shuffle within same identity
        assert cls_idx_ext_dict_1.keys() == cls_dix_ext_dict_2.keys()
        keys = cls_idx_ext_dict_1.keys()
        for key in keys:
            random.shuffle(cls_idx_ext_dict_1[key])
            random.shuffle(cls_dix_ext_dict_2[key])

        indices_1 = dict_value_to_list(cls_idx_ext_dict_1)
        indices_2 = dict_value_to_list(cls_dix_ext_dict_2)
        pairs = list(zip(indices_1, indices_2))
        self.pairs = pairs
        self.num_pairs = len(pairs)

        # 2. shuffle across identities
        random.shuffle(pairs)
        cnt = 0
        while cnt < len(pairs):  # true length go through training
            yield pairs[cnt]
            cnt += 1


if __name__ == '__main__':
    # from utils.helpers import makedir
    # train_ratio = 0.4  # todo: edit
    # split = 'split_{}_{}'.format(int(train_ratio * 100), int(100 - train_ratio * 100))
    # domains = ['Art', 'Clipart', 'Product', 'Real_World']
    # dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_org'
    # augment_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_aug'
    # annot_root_dir = '../data' + '/' + split

    ''' 1. Augmentation '''
    # for domain in domains:
    #     dataset_dir = os.path.join(dataset_root_dir, domain)
    #     dataset = BaseDataset(dataset_dir)
    #     dataset.augment(augment_root_dir)
    #     print('{} done'.format(domain))
    # print('done')

    ''' 2. Build Annotation File'''
    # for domain in domains:
    #     # original
    #     dataset_dir = os.path.join(dataset_root_dir, domain)
    #     dataset = BaseDataset(dataset_dir)
    #     dataset.get_annotations()
    #
    #     # augmented
    #     dataset_dir = os.path.join(augment_root_dir, domain + '_augmented')
    #     dataset = BaseDataset(dataset_dir)
    #     dataset.get_annotations()

    ''' 3. Create Train/Test Split'''
    # runs = 3
    # for run in range(runs):
    #     annot_dir = os.path.join(annot_root_dir, 'run_' + str(run))
    #     makedir(annot_dir)
    #
    #     for domain in domains:
    #         makedir(annot_dir + '/' + domain)
    #
    #         annot = os.path.join(dataset_root_dir, domain + '.txt')
    #         annot_aug = os.path.join(augment_root_dir, domain + '_augmented.txt')
    #         image_dataset = BaseImageDataset(annot=annot, annot_augmented=annot_aug, train_ratio=0.4)
    #
    #         image_dataset.split(shuffle=True)
    #         image_dataset.gen_annot_split(out_dir=annot_dir + '/' + domain)
    #
    #         annot_train = os.path.join(annot_dir + '/' + domain, domain + '_train.txt')
    #         image_dataset.gen_annot_split_aug(annot_train=annot_train, out_dir=annot_dir + '/' + domain)

    ''' Test Random Pair Sampler '''
    # dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_org'
    # annot_1 = '../data/split_40_60/run_0/Art/Art_train.txt'
    # annot_2 = '../data/split_40_60/run_0/Clipart/Clipart_train.txt'

    dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_aug'
    annot_1 = '../data/split_40_60/run_0/Art/Art_train_augmented.txt'
    annot_2 = '../data/split_40_60/run_0/Clipart/Clipart_train_augmented.txt'

    # book keeping namings and code
    from settings import img_size

    # load the data
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    # train set
    train_dataset = CrossDomainImageDataset(annot_1, annot_2, dataset_dir=dataset_root_dir, transform=transform)
    random_pair_sampler = RandomPairSampler(data_source=train_dataset)
    import torch
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=False,
        sampler=random_pair_sampler,
        num_workers=4, pin_memory=False, drop_last=False)

    epochs = 3
    for epoch in range(epochs):
        print('epoch: ', epoch)
        for i, (img_1, img_2, label_1, label_2) in enumerate(train_loader):
            # print(i, label_1, label_2)
            print('pairs = ', random_pair_sampler.num_pairs)
            transform = transforms.ToPILImage()
            if i == 2:
                img_1 = transform(img_1[1])
                img_1.show()
                img_2 = transform(img_2[1])
                img_2.show()
