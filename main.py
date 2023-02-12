import os
import shutil
import torch.utils.data
import torchvision.transforms as transforms
from datasets.bases import ImageDataset
import argparse
import re
from torchinfo import summary


from utils.helpers import makedir
from utils.logger import Logger
from models import model
from core import push
import train_test.train_and_test as tnt
from utils import save
from utils.log import create_logger
from datasets.preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='1')  # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-base_architecture', nargs=1, type=str, default='resnet50')
parser.add_argument('-domain', nargs=1, type=str, default='Art')
parser.add_argument('-run_id', default=0, type=int)
parser.add_argument('-train_ratio', default=0.8, type=float)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# book keeping namings and code
from settings import method, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, experiment_run

# load from args
base_architecture = args.base_architecture
train_ratio = args.train_ratio
domain = args.domain
run_id = args.run_id

# todo: uncomment this if using command line
# base_architecture = base_architecture[0]
# domain = domain[0]


base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
split = 'split_{}_{}'.format(int(train_ratio * 100), int(100 - train_ratio * 100))


model_dir = './saved_models/' + base_architecture + '/' + method + \
            '/' + split + '/' + 'run_{}'.format(run_id) + \
            '/' + domain + '/' + '/' + experiment_run + '/'
makedir(model_dir)

shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'feat', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_test', 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load data
from settings import train_batch_size, test_batch_size, train_push_batch_size
from settings import dataset_root_dir, dataset_augmented_root_dir, annot_root_dir

normalize = transforms.Normalize(mean=mean, std=std)

# all datasets
run = 'run_' + str(run_id)

# train set
mode = 'train_augmented'
annot_dir = annot_root_dir + '/' + split
train_dataset = ImageDataset(dataset_dir=dataset_augmented_root_dir,
                             annot=annot_dir + '/' + run + '/' + domain + '/' + domain + '_' + mode + '.txt',
                             transform=transforms.Compose([
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor(),
                                normalize,
                                ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                            num_workers=4, pin_memory=False)

# push set
mode = 'train'
train_push_dataset = ImageDataset(
                             dataset_dir=dataset_root_dir,
                             annot=annot_dir + '/' + run + '/' + domain + '/' + domain + '_' + mode + '.txt',
                             transform=transforms.Compose([
                                            transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                    ]))

train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# test set
mode = 'test'
test_dataset = ImageDataset(
                             dataset_dir=dataset_root_dir,
                             annot=annot_dir + '/' + run + '/' + domain + '/' + domain + '_' + mode + '.txt',
                             transform=transforms.Compose([
                                            transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            normalize,
                                ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

logger = Logger(model_dir + './log.txt', title=base_architecture_type)
logger.set_names(['epoch', 'train acc', 'test acc'])

logger_loss_train = Logger(model_dir + './log_loss_train.txt', title=base_architecture_type)
logger_loss_train.set_names(['epoch', 'cross_entropy', ' cluster ', 'separation', 'avg_separation', '\tl1\t', 'p_dist_pair'])

logger_loss_test = Logger(model_dir + './log_loss_test.txt', title=base_architecture_type)
logger_loss_test.set_names(['epoch', 'cross_entropy', ' cluster ', 'separation', 'avg_separation', '\tl1\t', 'p_dist_pair'])


# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
summary(ppnet, (1, 3, 224, 224))

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size

joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
     # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
     ]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs

warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
     ]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr

last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
best_acc = 0
log('start training')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    # No Push
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        train_acc, train_losses = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        train_acc, train_losses = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu, losses = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.2, log=log)
    if accu > best_acc:
        # torch.save(obj=model, f=os.path.join(model_dir, 'best_model_nopush.pth'))
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='best_model_nopush', accu=1,
                                    target_accu=best_acc, log=log)
        best_acc = accu
        print('****** Found better model. ******')

    logger.append([epoch, train_acc, accu])
    logger_loss_train.append([epoch,
                              train_losses['cross_entropy'], train_losses['cluster'], train_losses['separation'],
                              train_losses['avg_separation'], train_losses['l1'], train_losses['p_dist_pair']])
    logger_loss_test.append([epoch,
                              losses['cross_entropy'], losses['cluster'], losses['separation'],
                              losses['avg_separation'], losses['l1'], losses['p_dist_pair']])


    # Push
    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, losses = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.20, log=log)
        '''
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.20, log=log)
        '''
logclose()
logger.close()
