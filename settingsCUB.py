from datetime import datetime

# model
base_architecture = 'resnet50'

# data
img_size = 224
num_classes = 200


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

num_train_epochs = 50
num_warm_epochs = 5


push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
