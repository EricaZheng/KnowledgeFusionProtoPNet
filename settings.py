from datetime import datetime

# method
method = 'single'

# model
base_architecture = 'resnet50'
# options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# options: 'densenet121', 'densenet161', 'densenet169', 'densenet201'
# options: 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'

# data
img_size = 224
num_classes = 65

# dataset
domain = 'Art'  # options: 'Art', 'Clipart', 'Product', 'Real_World'
run_id = 0  # options: 0, 1, 2
train_ratio = 0.4  # options: 0.8, 0.4

dataset_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_org'
dataset_augmented_root_dir = '/home/bizon/Dataset/Office-Home/OfficeHomeDataset_aug'
annot_root_dir = './data'

# prototype
num_prototypes_per_class = 10
prototype_shape = (num_classes * num_prototypes_per_class, 128, 1, 1)

prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# output
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
experiment_run = dt_string

# train
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'crs_dom': 0.05,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}


num_train_epochs = 21  
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
