import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DeepLabV3 import deeplabv3_backbone
import numpy as np

from utils import *
from transforms_vision import *
from model_wrapper import ModelWrapper, ModelLoader
from dataset import CityscapesLoaders

def conf_to_mIoU(conf):
    num = [(conf[c,:].sum()+conf[:,c].sum()-conf[c,c]) for c in range(len(CITYSCAPES_LABELS))]
    IoUs = [(conf[c,c]/n if n else 1.0) for (c,n) in enumerate(num)]; return np.mean(IoUs[1:])

def IandU_to_mIoU(I, U):
    IoUs = [(i/u if u else 1.0) for (i,u) in zip(I, U)]; return np.mean(IoUs[1:])

class ISSModelWrapper(ModelWrapper):
    def __init__(self, net, **args):
        task_criterion = args.pop('task_criterion')
        super(ISSModelWrapper, self).__init__(net, **args)
        self.criterion = task_criterion.to(self.device)

    # For small batch size (2), batchnorm should be preserved in training mode
    def eval(self):
        self.net.train(); return self

    def run_batch(self, data, return_stats=False, return_outputs=False):
        X, M = data; T = F.interpolate(self(X)['out'], size=(M.shape[-2],M.shape[-1]), mode="bilinear", align_corners=True)
        loss = self.criterion(T.cpu(), M.cpu())
        if return_stats:
            P = T.detach().cpu().argmax(dim=1); M = M
            S = F.interpolate(X+.5, size=(M.shape[-2],M.shape[-1]), mode="bilinear", align_corners=True)[:,-1,:,:].cpu().long()
        return {
            'loss': loss,
            'outputs': T.detach().cpu() if return_outputs else None,
            'stats': {
                'loss': loss.item(),
                'pd_I': np.array([torch.logical_and(torch.eq(M,c),torch.eq(P,c)).sum().item() for c in [0,1]]),
                'pd_U': np.array([torch.logical_or (torch.eq(M,c),torch.eq(P,c)).sum().item() for c in [0,1]]),
                'sg_I': np.array([torch.logical_and(torch.eq(M,c),torch.eq(S,c)).sum().item() for c in [0,1]]),
                'sg_U': np.array([torch.logical_or (torch.eq(M,c),torch.eq(S,c)).sum().item() for c in [0,1]]),
                # 'conf': confusion_matrix(y_true=M, y_pred=P, labels=[0,1]),
            } if return_stats else None,
        }
    
    def fit(self, loaders, num_epochs=100, gradient_step=16, validate_per=1, save_per=10):
        for _ in range(num_epochs):
            result = self.run_epoch(loaders, gradient_step=gradient_step, validate_per=validate_per, save_per=save_per)
            stats = dict(**result['train_stats'],**result['valid_stats'])
            stats['train_pd_mIoU'] = IandU_to_mIoU(stats['train_pd_I'], stats['train_pd_U']); stats.pop('train_pd_I'); stats.pop('train_pd_U')
            stats['valid_pd_mIoU'] = IandU_to_mIoU(stats['valid_pd_I'], stats['valid_pd_U']); stats.pop('valid_pd_I'); stats.pop('valid_pd_U')
            stats['train_sg_mIoU'] = IandU_to_mIoU(stats['train_sg_I'], stats['train_sg_U']); stats.pop('train_sg_I'); stats.pop('train_sg_U')
            stats['valid_sg_mIoU'] = IandU_to_mIoU(stats['valid_sg_I'], stats['valid_sg_U']); stats.pop('valid_sg_I'); stats.pop('valid_sg_U')
            if self.console_log:
                self.logger.info(
                    "[%s] Epoch #%04d Result | train_loss %7.4f | train_mIoU %6.2f%% (%+6.2f%%) | valid_loss %7.4f | valid_mIoU %6.2f%% (%+6.2f%%)"%
                    (self.model_name,self.epoch,stats['train_loss'],stats['train_pd_mIoU']*100,stats['train_pd_mIoU']*100-stats['train_sg_mIoU']*100,stats['valid_loss'],stats['valid_pd_mIoU']*100,stats['valid_pd_mIoU']*100-stats['valid_sg_mIoU']*100)
                )
            updated = False
            for key, value in stats.items():
                if self.tracker.update(key, value, self.epoch) and key==self.primary:
                    updated = True
            if updated:
                self.save("best.pth")
                if self.console_log:
                    self.logger.info("Best mIoU -> %6.2f%% at Epoch #%04d! Model Saved!"%(stats['valid_pd_mIoU']*100,self.epoch))
                    self.logger.info(ViewDictS(self.tracker.profile()))

MODEL_BACKBONE_MAPPING = {
    'ResNet50': 'resnet50',
    'DenseNet121': 'densenet121',
    'ResNeSt50': 'resnest50d',
    'HRNetW48': 'hrnet_w48',
}
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_backbone', dest='model_backbone', choices=['ResNet50','DenseNet121','ResNeSt50','HRNetW48'], default='HRNetW48', help="type of model to use")
    parser.add_argument('-cross', dest='K', type=int, default=0, help="cross validation's fold number (<= 0 for no cross validation)")
    parser.add_argument('-fold', dest='k', type=int, default=0, help="cross validation's fold index (0 <= index < cross_validation)")
    parser.add_argument('-epoch', dest='epoch', type=int, default=175, help="number of epochs to train")
    parser.add_argument('-lr', dest='lr', type=float, default=5e-4, help="learning rate (real lr = lr * step)")
    parser.add_argument('-step', dest='step', type=int, default=8, help="gradient accumulation step")
    parser.add_argument('--restart', dest='restart', action='store_const', const=True, default=False, help='clear previous trained model')
    parser.add_argument('--debug', dest='debug', action='store_const', const=True, default=False, help='debug mode: small dataset')
    parser.add_argument('--ddp', dest='ddp', action='store_const', const=True, default=False, help='use distributed data parallel')
    parser.add_argument('--ddp_gpu', default="0", type=str, help='gpu indices')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()

    net = deeplabv3_backbone(
        backbone_type = MODEL_BACKBONE_MAPPING[args.model_backbone],
        num_classes = 2,
        extra_input_channel = 1,
    )
    optimizer = optim.Adam(
        net.parameters(),
        lr = args.lr,
        weight_decay = 2e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=(60,120,150),gamma=0.3)
    if args.ddp:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.ddp_gpu
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = 'cuda:0'
    model = ISSModelWrapper(
        net,
        model_root = "models/",
        model_name = "DeepLabV3-"+args.model_backbone+"-IIS",
        model_version = "fold-%1d"%args.k,
        optimizer = optimizer,
        scheduler = scheduler,
        tracker = [
            ('epoch','train_loss','less'),
            ('epoch','valid_loss','less'),
            ('epoch','train_pd_mIoU','greater'),
            ('epoch','valid_pd_mIoU','greater'),
            ('epoch','train_sg_mIoU','greater'),
            ('epoch','valid_sg_mIoU','greater'),
        ],
        task_criterion = nn.CrossEntropyLoss(),
        primary = 'valid_pd_mIoU',
        ddp = args.ddp,
        device = device,
        clear = args.restart,
    )
    loaders = CityscapesLoaders(
        mode = 'fine',
        transforms = {
            'train': SUGGEST_TRANSFORM,
            'valid': SUGGEST_TRANSFORM,
            'testi': SUGGEST_TRANSFORM,
        },
        batch_sizes = {
            'train': 2*(args.ddp_gpu.count(',')+1),
            'valid': 2*(args.ddp_gpu.count(',')+1),
            'testi': 2*(args.ddp_gpu.count(',')+1),
        },
        num_workers = 16,
        cross_validation = args.K,
        fold = args.k,
        ddp = args.ddp,
        debug = args.debug,
    )

    model.fit(loaders, num_epochs=args.epoch)