import os
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model_90k import CNN3Dver3_90k
from DataSet import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2
from train_and_eval import train_one_epoch, evaluate,plot_image, WeightedMSELoss

random.seed(26)
np.random.seed(26)
torch.manual_seed(26)
torch.cuda.manual_seed(26)
torch.cuda.manual_seed_all(26) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或者 ":4096:8"

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    tb_writer = SummaryWriter(log_dir="runs/CNN3Dver3_90k/Demo0")
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
                        split_shuffle = False,
                        transform=data_transform["without_jet"])
    train_dataset = dataset_2(data_set.train_X, data_set.train_Y)
    val_dataset = dataset_2(data_set.val_X, data_set.val_Y)
    test_dataset = dataset_2(data_set.test_X, data_set.test_Y)
    data_set_jet = MyDataSet(img_dir=args.jet_dir,
                                    group_size=1000,
                                    size_in = 1000,
                                    splition= False,
                                    split_shuffle = False,
                                    transform=data_transform["jet"])
    jet_dataset = dataset_2(data_set_jet.X, data_set_jet.Y)
    
    batch_size = args.batch_size
    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers
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
    
    jet_loader = torch.utils.data.DataLoader(jet_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    # 实例化模型
    model = CNN3Dver3_90k().to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 4, 56, 56), device=device)
    tb_writer.add_graph(model, init_img)

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
        
    warmup_epochs_1 = 40
    warmup_epochs_2 = 80
    warmup_epochs_3 = 83
    learningrate = args.lr

    def lf_function(epoch): 
        if epoch < warmup_epochs_1:
            return 1
        elif epoch < warmup_epochs_2: 
            return 0.1
        elif epoch < warmup_epochs_3:
            return((epoch - warmup_epochs_2) / (warmup_epochs_3 - warmup_epochs_2)) * 0.5 + 0.1
        else:
            return(((1 + math.cos((epoch - warmup_epochs_3) * math.pi / (args.epochs - warmup_epochs_3))) / 2) * 0.5 + 0.1)
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_function)
    loss_function_val = torch.nn.MSELoss()
    loss_function_test = torch.nn.MSELoss()
    # loss_function_train = WeightedMSELoss(alpha=0.5, reduction='mean')
    loss_function_train = torch.nn.MSELoss()
    
    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    loss_function=loss_function_train)
        # update learning rate
        scheduler.step()

        # validate
        if args.patten == "train":
            test_loss = evaluate(model=model,
                    data_loader=val_loader,
                    device=device,
                    loss_function=loss_function_val)
        else:
            test_loss = evaluate(model=model,
                    data_loader=test_loader,
                    device=device,
                    loss_function=loss_function_test)

        # add loss, acc and lr into tensorboard
        print("[epoch {}] loss: {}".format(epoch, round(test_loss, 7)))
        tags = ["train_loss", "test_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], test_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # add figure into tensorboard
        if (epoch + 1) % 10 == 0:
            fig_test = plot_image(net = model, 
                                data_loader = val_loader,
                                device = device,
                                label = "test")
            fig_jet = plot_image(net = model,
                                data_loader = jet_loader,
                                device = device,
                                label = "jet")

            if fig_test is not None:
                tb_writer.add_figure("predictions without jet",
                                    figure=fig_test,
                                    global_step=epoch)
            if fig_jet is not None:
                tb_writer.add_figure("predictions with jet",
                                    figure=fig_jet,
                                    global_step=epoch)
    torch.save(model.state_dict(), "./weights/CNN3Dver3_90k.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--saving_routine', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patten', type=str, default="Parameter")
    
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