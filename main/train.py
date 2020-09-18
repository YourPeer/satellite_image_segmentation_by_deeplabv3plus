#-*-conding:utf-8-*-
import sys
sys.path.append("../")
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data import BasicDataset
from model import deeplabv3plus_resnet50,deeplabv3plus_mobilenetv2
import os
from eval_net import eval_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def generate_dataloader(train_lst_dir,val_lst_dir,batch_size=64, num_workers=8):
    sate_dataset_train = BasicDataset(train_lst_dir)  # ?????????????????
    sate_dataset_val = BasicDataset(val_lst_dir)
    train_dataloader = DataLoader(sate_dataset_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)  # ???????data_loader
    eval_dataloader = DataLoader(sate_dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 drop_last=True)  # ???????data_loader?drop_last??????batch??32???
    return train_dataloader,eval_dataloader,len(sate_dataset_train)




def save_model(net,fw_iou_avg,epoch):
    if os.path.exists(dir_checkpoint+checkpoint_name):
        checkpoint = torch.load(dir_checkpoint + checkpoint_name)
        print(fw_iou_avg, checkpoint['fw_iou_avg'])
        if fw_iou_avg > checkpoint['fw_iou_avg']:
            print('save better model!')
            state = {'net': net.state_dict(), 'fw_iou_avg': fw_iou_avg, 'epochth': epoch + 1}
            torch.save(state, dir_checkpoint + checkpoint_name)
            print(f'checkpoint {epoch + 1} saved!')
    else:
        state = {'net': net.state_dict(), 'fw_iou_avg': fw_iou_avg,'epochth': epoch + 1}
        torch.save(state, dir_checkpoint + checkpoint_name)
        print(f'checkpoint {epoch + 1} saved!')

def eval_one_epoch(net,eval_dataloader,device,global_step,fw_iou_avg,writer):
    val_loss, pixel_acc_avg, mean_iou_avg, _fw_iou_avg = eval_net(net, eval_dataloader, device)
    if fw_iou_avg < _fw_iou_avg:
        fw_iou_avg = _fw_iou_avg
    logging.info('Validation cross entropy: {}'.format(val_loss))
    writer.add_scalar('Loss/test', val_loss, global_step)
    writer.add_scalar('pixel_acc_avg', pixel_acc_avg, global_step)
    writer.add_scalar('mean_iou_avg', mean_iou_avg, global_step)
    writer.add_scalar('fw_iou_avg', fw_iou_avg, global_step)
    return fw_iou_avg

def train_one_epoch(net,criterion,optimizer,scheduler,writer,
                    train_dataloader,eval_dataloader,train_steps,epoch,epochs,
                    device,batch_size,global_step,fw_iou_avg):
    epochs_loss=0

    with tqdm(total=train_steps, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for idx, batch_samples in enumerate(train_dataloader):
            batch_image, batch_mask = batch_samples["image"], batch_samples["mask"]
            batch_image = batch_image.to(device=device, dtype=torch.float32)
            logits = net(batch_image)  # torch.Size([batchsize, 8, 256, 256])
            y_true = batch_mask.to(device=device, dtype=torch.long)  # torch.Size([batchsize, 256, 256])
            loss = criterion(logits, y_true)
            epochs_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)  # ????
            optimizer.step()
            pbar.update(batch_image.shape[0])  # ???????????10
            scheduler.step(loss)  # ?????????
            global_step += 1
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            print(train_steps)
            if global_step % (train_steps // ( 1*batch_size)) == 0:
                fw_iou_avg=eval_one_epoch(net,eval_dataloader,device,global_step,fw_iou_avg,writer)
        return fw_iou_avg

def train(net,checkpoint_name,batchsize,lr,epochs,device,train_lst_dir,val_lst_dir):
    global_step = 0
    train_dataloader,val_dataloader,train_steps=generate_dataloader(train_lst_dir,val_lst_dir,batch_size=64, num_workers=8)
    criterion = nn.CrossEntropyLoss().to(device)  # ???????
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)  # ???
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=120, factor=0.95,
                                                     min_lr=5e-5)  # ??????
    writer = SummaryWriter(comment=checkpoint_name + f'LR_{lr}_BS_{batchsize}')  # ????tensorboard??
    fw_iou_avg = 0
    for epoch in range(epochs):
        fw_iou_avg=train_one_epoch(net, criterion, optimizer, scheduler, writer,
                        train_dataloader, val_dataloader, train_steps, epoch, epochs,
                        device, batchsize, global_step,fw_iou_avg)

        save_model(net,fw_iou_avg,epoch)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batchsize', metavar='B', type=int, nargs='?', default=256,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--lr', metavar='LR', type=float, nargs='?', default=5e-4,
                        help='Learning rate', dest='lr')
    return parser.parse_args()

def main():

    args = get_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = deeplabv3plus_mobilenetv2()
    net.to(device=device)
    print(f'Using device {device}')
    train_lst_dir, val_lst_dir="../data/train.lst","../data/val.lst"
    train(
        net, checkpoint_name,
        args.batchsize, args.lr, args.epochs, device,
        train_lst_dir, val_lst_dir
    )

if __name__=="__main__":
    dir_checkpoint = '../checkpoints/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    checkpoint_name = 'deeplabv3plus.pth'
    main()