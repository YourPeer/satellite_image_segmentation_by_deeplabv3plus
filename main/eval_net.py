import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 21*21???,???ground truth??,???preds???,???

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)  # ??0??mean,shape:[21]
        return MIoU

    def Class_IOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # --------------------------------------------------------------------------------
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def eval_net(net,data_loader,device):
    net.eval()
    val_batch_num=len(data_loader)
    eval_loss=0

    e = Evaluator(num_class=8)
    pixel_acc_avg = 0
    mean_iou_avg = 0
    fw_iou_avg = 0

    with tqdm(total=val_batch_num, desc='Validation round', unit='batch', leave=False) as pbar:
        for idx,batch_samples in enumerate(data_loader):
            batch_image, batch_mask = batch_samples["image"], batch_samples["mask"]
            batch_image=batch_image.to(device=device,dtype=torch.float32)
            mask_true=batch_mask.to(device=device,dtype=torch.long)

            with torch.no_grad():
                mask_pred=net(batch_image)
                probs = F.softmax(mask_pred, dim=1).squeeze(0)  # [8, 256, 256]
                pre = torch.argmax(probs, dim=1)  # [256,256]

            e.add_batch(mask_true.cpu().data.numpy(),pre.cpu().data.numpy())
            pixel_acc=e.Pixel_Accuracy()
            pixel_acc_avg+=pixel_acc

            mean_iou=e.Mean_Intersection_over_Union()
            mean_iou_avg+=mean_iou

            fw_iou=e.Frequency_Weighted_Intersection_over_Union()
            fw_iou_avg+=fw_iou

            eval_loss+=F.cross_entropy(mask_pred,mask_true).item()
            pbar.set_postfix({'eval_loss': eval_loss/(idx+1)})
            pbar.update()
            e.reset()
    print("pixel_acc_avg:"+str(pixel_acc_avg/val_batch_num))
    print("mean_iou_avg:" + str(mean_iou_avg / val_batch_num))
    print("fw_iou_avg:" + str(fw_iou_avg / val_batch_num))
    net.train()
    return eval_loss/val_batch_num,pixel_acc_avg/val_batch_num,mean_iou_avg / val_batch_num,fw_iou_avg / val_batch_num



