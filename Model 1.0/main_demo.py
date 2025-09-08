import os
from tqdm import tqdm
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import CNN3D
from DataSet import MaxMinNormalizeGlobalPerChannel, MyDataSet, dataset_2
from train_and_eval import train_one_epoch, evaluate


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 定义训练以及预测时的预处理方法
    data_transform = {
        "without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
        "jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}

    # 实例化训练数据集
    data_set = MyDataSet(img_dir=args.img_dir,
                        group_size=10000,
                        size_in = 10000,
                        splition = True,
                        transform=data_transform["without_jet"])
    train_dataset = dataset_2(data_set.train_X, data_set.train_Y)
    val_dataset = dataset_2(data_set.val_X, data_set.val_Y)
    test_dataset = dataset_2(data_set.test_X, data_set.test_Y)
    
    batch_size = args.batch_size
    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw)

    # 实例化模型
    model = CNN3D().to(device)
    # 如果存在预训练权重则载入
    if args.weights is None:
        print("No weights file provided. Using random defaults.")
    else:
        model.load_state_dict(torch.load(args.weights))
        print("using pretrain-weights.")
    
    # 是否冻结权重
    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "decoder" not in name:
                para.requires_grad_(False)
    optimizer = optim.Adam(model.parameters() ,lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.005)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    loss_function = nn.MSELoss()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    loss_function=loss_function)
        scheduler.step()

        # validate
        if args.patten == "train":
            test_loss = evaluate(model=model,
                    data_loader=val_loader,
                    device=device,
                    loss_function=loss_function)
        else:
            test_loss = evaluate(model=model,
                    data_loader=test_loader,
                    device=device,
                    loss_function=loss_function)

        print("[epoch {}] loss: {}".format(epoch, round(test_loss, 7)))
        if ((epoch+1) % args.saving_routine == 0) or (epoch == args.epochs-1):
            # save weights
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--saving_routine', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patten', type=str, default="train")
    
    img_root = 'Gauss_S1.00_NL0.30_B0.50\Gauss_S1.00_NL0.30_B0.50'
    parser.add_argument('--img_dir', type=str, default=img_root)
    jet_root = 'Gauss_S1.00_NL0.30_B0.50_Jet\Gauss_S1.00_NL0.30_B0.50_Jet'
    parser.add_argument('--jet_dir', type=str, default=jet_root)

    parser.add_argument('--weights', type=str, default= None,
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    train(opt)