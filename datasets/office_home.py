import os
from .bases import BaseImageDataset


class OfficeHome(BaseImageDataset):
    def __init__(self, dataset_root_dir, augment_root_dir, verbose=True):
        super(BaseImageDataset).__init__()
        self.domains = ['Art', 'Clipart', 'Product', 'Real_World']
        self.dataset_root_dir = dataset_root_dir
        self.augment_root_dir = augment_root_dir
        self.verbose = verbose

    def _check_before_run(self):
        pass

    def build(self, runs=1):
        """
        Loop over all domains
            generate NEW train/test split (shuffle=True) !!!
            save in <run_1>, <run_2>, ...

        Example:
            output
                /run_1
                    Art_train.txt
                    Art_test.txt
                    Art_train_augmented.txt
                /run_2
                    Art_train.txt
                    Art_test.txt
                    Art_train_augmented.txt
                ...
        """
        if self.verbose:
            print('[info] create xxx')

    def print_data_statistics(self):
        pass

    def get_annotations(self, annotation_dir=None):
        for domain in self.domains:
            dataset_dir = os.path.join(self.dataset_root_dir, domain)
            self.augment(self.dataset_root_dir)
            print('{} done'.format(domain))
        print('done')

        for domain in self.domains:
            dataset_dir = os.path.join(dataset_root_dir, domain)
            self.get_annotations(annotation_dir)

            dataset_dir = os.path.join(self.augment_root_dir, domain + '_augmented')
            self.get_annotations(annotation_dir)



