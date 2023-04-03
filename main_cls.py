import argparse
import os
import time

import mindspore as ms
import mindspore.dataset as ds

from mindspore import nn
from mindspore.ops import value_and_grad
from mindspore.nn.metrics import Accuracy
from mindspore.communication import init, get_rank
from src.dataset import ModelNet40
from src.models import DGCNN_cls, NLLLoss
from src.provider import RandomInputDropout


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('MindSpore DGCNN Training Configurations.')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N', choices=['pointnet', 'dgcnn'], help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='step', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--num_workers', type=int, default=4, help='num of worker to Dataloader')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

    parser.add_argument('--data_path', type=str, default='/home/root/code_demo/dgcnn.pytorch/data/', help='data path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='/home/root/code_demo/dgcnn_mindspore/save', help='save root')

    parser.add_argument('--platform', type=str, default='Ascend', help='run platform')

    return parser.parse_known_args()[0]


def content_init(args, device_id, device_num):
    """content_init"""
    _platform = args.platform
    _platform = _platform.lower()

    if not _platform in ("ascend", "gpu", "cpu"):
        raise ValueError("Unsupported platform {}".format(args.platform))

    if _platform == "ascend":
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
        # ms.set_context(device_target="Ascend", device_id=device_id)
        ms.set_context(max_call_depth=2048)

        if device_num > 1:
            init()
            ms.set_auto_parallel_context(parallel_mode="data_parallel", gradients_mean=True)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", max_call_depth=2048)
        if device_num > 1:
            ms.dataset.config.set_enable_shared_mem(False)
            ms.set_auto_parallel_context( parallel_mode="data_parallel", gradients_mean=True, device_num=device_num)
            ms.set_seed(1234)
            init()
        else:
            ms.set_context(device_id=device_id)


def get_data_url(args):
    '''get_data_url'''
    log_file = None
    local_data_url = args.data_path
    pretrained_ckpt_path = args.pretrained_ckpt if args.pretrained_ckpt.endswith('.ckpt') else ""
    local_train_url = args.save_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return local_data_url, pretrained_ckpt_path, local_train_url, log_file


def get_train_ds(device_num, train_ds_generator, num_workers, rank_id):
    """get_train_ds"""
    if device_num > 1:
        train_ds = ds.GeneratorDataset(train_ds_generator, ["data", "label"],
                                       num_parallel_workers=num_workers,
                                       shuffle=True,
                                       shard_id=rank_id,
                                       num_shards=device_num)
    else:
        train_ds = ds.GeneratorDataset(train_ds_generator, ["data", "label"],
                                       num_parallel_workers=num_workers,
                                       shuffle=True)
    return train_ds


def train(args):
    # INIT
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    content_init(args, device_id, device_num)
    rank_id = get_rank() if device_num > 1 else device_id
    local_data_url, pretrained_ckpt_path, local_train_url, log_file = get_data_url(args)
    print(f"device:{device_num}, rank_id:{rank_id}")
    print(args)
    # DATA LOADING
    print('Load dataset ...')
    # train
    train_ds_generator = ModelNet40(num_points=args.num_points, data_dir=local_data_url, partition='train')
    train_ds = get_train_ds(device_num, train_ds_generator, args.num_workers, rank_id)
    random_input_dropout = RandomInputDropout()
    train_ds = train_ds.batch(batch_size=args.batch_size,
                              drop_remainder=True,
                              per_batch_map=random_input_dropout,
                              input_columns=["data", "label"],
                              num_parallel_workers=args.num_workers,
                              python_multiprocessing=True)
    steps_per_epoch = train_ds.get_dataset_size()
    # test
    test_ds_generator = ModelNet40(num_points=args.num_points, data_dir=local_data_url, partition='test')
    test_ds = get_train_ds(device_num, test_ds_generator, args.num_workers, rank_id)
    test_ds = test_ds.batch(batch_size=args.batch_size, drop_remainder=True, num_parallel_workers=args.num_workers)

    # MODEL
    if args.model == 'pointnet':
        net = PointNet(args)
    elif args.model == 'dgcnn':
        net = DGCNN_cls(args)
    else:
        raise Exception("Not implemented")
    print(str(net))

    # load checkpoint
    if args.pretrained_ckpt.endswith('.ckpt'):
        print("Load checkpoint: %s" % args.pretrained_ckpt)
        param_dict = ms.load_checkpoint(pretrained_ckpt_path)
        ms.load_param_into_net(net, param_dict)

    if args.use_sgd:
        print("Use SGD")
        net_opt = nn.SGD(net.trainable_params(), learning_rate=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        net_opt = nn.Adam(net.trainable_params(), learning_rate=args.lr, weight_decay=1e-4)

    # lr_epochs = list(range(20, args.epochs+1, 20))
    # lr = ms.numpy.linspace(0.001, 0.0005, len(lr_epochs))
    # if args.scheduler == 'cos':
    #     # scheduler = nn.cosine_decay_lr(opt, args.epochs, eta_min=1e-3)
    #     stop = 0
    # elif args.scheduler == 'step':
    #     scheduler = nn.piecewise_constant_lr(lr_epochs, learning_rates=lr)
    
    # net_loss = NLLLoss()
    net_loss = nn.NLLLoss()
    model = ms.Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()}) 
    config = ms.CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    ckpt_callback = ms.ModelCheckpoint(prefix='DGCNN', directory=local_train_url, config=config)
    # loss_callback = ms.LossMonitor(steps_per_epoch)

    call_back = []
    call_back += [ms.TimeMonitor()]
    call_back += [ms.LossMonitor(50)]
    call_back += [ckpt_callback]

    # TRAINING
    net.set_train()
    print('Starting training ...')
    print('Time: ', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    time_start = time.time()
    # model.train(epoch=args.epochs,
    #             train_dataset=train_ds,
    #             callbacks=call_back,
    #             dataset_sink_mode=False)
    model.fit(args.epochs, train_ds, test_ds, callbacks=call_back)
    
    # END
    print('End of training.')
    print('Total time cost: {} min'.format("%.2f" %((time.time() - time_start) / 60)))

    # TRAINING 2
    # for i in range(args.epochs):
    #     print(f"Epoch {i+1}\n-------------------------------")

    #     def train_loop(model, dataset, loss_fn, optimizer):
    #         # Define forward function
    #         def forward_fn(data, label):
    #             logits = model(data)
    #             loss = loss_fn(logits, label)
    #             return loss, logits
    #         # Get gradient function
    #         grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    #         # Define function of one-step training
    #         def train_step(data, label):
    #             (loss, _), grads = grad_fn(data, label)
    #             loss = ops.depend(loss, optimizer(grads))
    #             return loss
    #         size = dataset.get_dataset_size()
    #         model.set_train()
    #         for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
    #             loss = train_step(data, label)
    #             if batch % 100 == 0:
    #                 loss, current = loss.asnumpy(), batch
    #                 print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


    #     train_loop(net, train_ds, net_loss, net_opt)
    # print("Done!")


if __name__ == "__main__":
    args = parse_args()
    if not args.eval:
        train(args)
    else:
        test(args)
