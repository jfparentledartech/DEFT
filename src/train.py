from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from torch.utils import bottleneck
from lib.opts import opts
from lib.model.model import create_model, load_model, save_model
from lib.model.data_parallel import DataParallel
from lib.logger import Logger
from lib.dataset.dataset_factory import get_dataset
from lib.trainer import Trainer
import warnings

import pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_optimizer(opt, model):
    if opt.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == "sgd":
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001
        )
    else:
        assert 0, opt.optim
    return optimizer


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    if not opt.not_set_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.device = torch.device("cuda" if opt.gpus[0] >= 0 else "cpu")
    logger = Logger(opt)

    # print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    if opt.load_model != "":
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, opt, optimizer
        )

    for i, param in enumerate(model.parameters()):
        param.requires_grad = True
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.val_intervals < opt.num_epochs or opt.test:

        print("Setting up validation data...")
        if opt.use_subset:
            valset = Dataset(opt, "val")

            val_mask = list(range(0, int(len(valset) / 10), 1))
            val_subset = torch.utils.data.Subset(Dataset(opt, "val"), val_mask)

            print(len(val_subset))

            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True
            )

        else:
            val_loader = torch.utils.data.DataLoader(
                Dataset(opt, "val"),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=True
            )

        if opt.test:
            _, preds = trainer.val(0, val_loader)
            val_loader.dataset.run_eval(preds, opt.save_dir)
            return

    if opt.use_subset:
        print("Setting up train data...")

        trainset = Dataset(opt, "train")

        train_mask = list(range(0, int(len(trainset)/10), 1))
        train_subset = torch.utils.data.Subset(Dataset(opt, "train"), train_mask)

        print(len(train_subset))

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            Dataset(opt, "train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )


    print("Starting training...")
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        save_model(
            os.path.join(opt.save_dir, "model_{}.pth".format(epoch)),
            epoch,
            model,
            optimizer,
        )
        mark = epoch if opt.save_all else "last"
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write("epoch: {} |".format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary("train_{}".format(k), v, epoch)
            logger.write("{} {:8f} | ".format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(
                os.path.join(opt.save_dir, "model_{}.pth".format(mark)),
                epoch,
                model,
                optimizer,
            )
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                if opt.eval_val:
                    val_loader.dataset.run_eval(preds, opt.save_dir, epoch)
            for k, v in log_dict_val.items():
                logger.scalar_summary("val_{}".format(k), v, epoch)
                logger.write("{} {:8f} | ".format(k, v))
        else:
            save_model(
                os.path.join(opt.save_dir, "model_last.pth"), epoch, model, optimizer
            )
        logger.write("\n")
        #     if epoch in opt.save_point:
        save_model(
            os.path.join(opt.save_dir, "model_{}.pth".format(epoch)),
            epoch,
            model,
            optimizer,
        )
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print("Drop LR to", lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()

    filename = '../options/train_opt_pixset.txt'
    with open(filename, 'wb') as f:
        pickle.dump(opt, f)
    #     print(f'saved {filename}')
    # with open(filename, 'rb') as f:
    #     opt = pickle.load(f)
    # opt.resume = True
    # opt.use_pixell = True
    # opt.eval_val = True
    print(f'Using pixell -> ', opt.use_pixell)
    print(f'Using lstm -> ', opt.lstm)
    print(f'Sensor dropout -> ', opt.sensor_dropout)
    print(f'Using subset -> ', opt.use_subset)
    main(opt)
