import lightnet as ln
import torch

__all__ = ['params']


params = ln.engine.HyperParameters( 
    # Network
    class_label_map = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    _input_dimension = (416, 416),
    _batch_size = 64,
    _mini_batch_size = 8,
    _max_batches = 80200,

    # Dataset
    _train_set = 'train.h5',
    _test_set = 'test.h5',
    _filter_anno = 'ignore',

    # Data Augmentation
    _jitter = .3,
    _flip = .5,
    _hue = .1,
    _saturation = 1.5,
    _value = 1.5,
)

# Network
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

params.network = ln.models.TinyYolo(len(params.class_label_map))
params.network.apply(init_weights)

# Loss
params.loss = ln.network.loss.RegionLoss(
    len(params.class_label_map),
    params.network.anchors,
    params.network.stride,
)

# Postprocessing
params._post = ln.data.transform.Compose([
    ln.data.transform.GetBoundingBoxes(len(params.class_label_map), params.network.anchors, 0.001),
    ln.data.transform.NonMaxSuppression(0.5),
    ln.data.transform.TensorToBrambox(params.input_dimension, params.class_label_map),
])

# Optimizer
params.optimizer = torch.optim.SGD(
    params.network.parameters(),
    lr = .001 / (params.batch_size // params.mini_batch_size),
    momentum = .9,
    weight_decay = .0005 * (params.batch_size // params.mini_batch_size),
    dampening = 0,
)

# Scheduler
burn_in = torch.optim.lr_scheduler.LambdaLR(
    params.optimizer,
    lambda b: (b / 1000) ** 4,
)
step = torch.optim.lr_scheduler.MultiStepLR(
    params.optimizer,
    milestones = [40000, 60000],
    gamma = .1,
)
params.scheduler = ln.engine.SchedulerCompositor(
#   batch   scheduler
    (0,     burn_in),
    (1000,  step),
)
