import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from sklearn.metrics import confusion_matrix
import numpy as np

from utils import *
from model_wrapper import ModelWrapper, ModelLoader
from dataset import CityscapesLoaders, CITYSCAPES_IMAGE_SIZE, CITYSCAPES_REDUCED_IMAGE_SIZE

# From https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/f9fb1ba66ff8aea29d833b885f08df64e62c2b23/lib/datasets/cityscapes.py
CITYSCAPES_CLASS_WEIGHT = torch.FloatTensor([0.2607858807557244, 0.11105729594313882, 0.4631900657413155, 0.16478468015176612, 1.2964421903686474, 1.2043504786235877, 1.0851663592999017, 1.531315380178946, 1.3427096764756152, 0.22147498684488204, 1.1063310921684824, 0.6113905232204833, 1.0886365551600967, 1.57419046993965, 0.41951892566348514, 1.4907370845843664, 1.5107508249316228, 1.5107508249316228, 1.5965511282395164, 1.4098655767771506])
# End From
CITYSCAPES_LABELS_LIST = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
CITYSCAPES_LABELS = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

def conf_to_mIoU(conf):
    num = [(conf[c,:].sum()+conf[:,c].sum()-conf[c,c]) for c in range(len(CITYSCAPES_LABELS))]
    IoUs = [(conf[c,c]/n if n else 1.0) for (c,n) in enumerate(num)]; return np.mean(IoUs[1:])

def IandU_to_mIoU(I, U):
    IoUs = [(i/u if u else 1.0) for (i,u) in zip(I, U)]; return np.mean(IoUs[1:])

class ISSModelWrapper(ModelWrapper):
    def __init__(self, net, task_criterion, **args):
        super(ISSModelWrapper, self).__init__(net, **args)
        self.criterion = task_criterion.to(self.device)

    def run_batch(self, data, return_stats=False, return_output=False):
        X, M = data; T = F.interpolate(self(X)['out'], size=(M.shape[-2],M.shape[-1]), mode="bilinear", align_corners=True)
        loss = self.criterion(T, M.to(self.device))
        if return_stats:
            P = T.detach().argmax(dim=1)
            M = M.to(self.device)
        return {
            'loss': loss,
            'output': T.detach().cpu() if return_output else None,
            'stats': {
                'loss': loss.item(),
                'I': np.array([torch.logical_and(torch.eq(M,c),torch.eq(P,c)).sum().item() for c in CITYSCAPES_LABELS_LIST]),
                'U': np.array([torch.logical_or (torch.eq(M,c),torch.eq(P,c)).sum().item() for c in CITYSCAPES_LABELS_LIST]),
                # 'conf': confusion_matrix(y_true=M, y_pred=P, labels=CITYSCAPES_LABELS_LIST),
            } if return_stats else None,
        }
    
    def fit(self, loaders, num_epochs=100, gradient_step=16, validate_per=1, save_per=None):
        for _ in range(num_epochs):
            result = self.run_epoch(loaders, gradient_step=gradient_step, validate_per=validate_per, save_per=save_per)
            stats = dict(**result['train_stats'],**result['valid_stats'])
            stats['train_mIoU'] = IandU_to_mIoU(stats['train_I'], stats['train_U']); stats.pop('train_I'); stats.pop('train_U')
            stats['valid_mIoU'] = IandU_to_mIoU(stats['valid_I'], stats['valid_U']); stats.pop('valid_I'); stats.pop('valid_U')
            self.logger.info(
                "[%s] Epoch #%04d Result | train_loss %7.4f | train_mIoU %6.2f%% | valid_loss %7.4f | valid_mIoU %6.2f%%"%
                (self.model_name,self.epoch,stats['train_loss'],stats['train_mIoU']*100,stats['valid_loss'],stats['valid_mIoU']*100)
            )
            for key, value in stats.items():
                if self.tracker.update(key, value, self.epoch) and key==self.primary:
                    self.save("best.pth"); self.logger.info("Best mIoU -> %6.2f%% at Epoch #%04d! Model Saved!"%(stats['valid_mIoU']*100,self.epoch))
                    self.logger.info(ViewDictS(self.tracker.profile()))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', dest='model_name', choices=['DeepLabV3'], default='DeepLabV3', help="type of model to use")
    parser.add_argument('-cross', dest='K', type=int, default=5, help="cross validation's fold number (<= 0 for no cross validation)")
    parser.add_argument('-fold', dest='k', type=int, default=0, help="cross validation's fold index (0 <= index < cross_validation)")
    parser.add_argument('-epoch', dest='epoch', type=int, default=175, help="number of epochs to train")
    parser.add_argument('-lr', dest='lr', type=float, default=2e-4, help="learning rate (real lr = lr * step)")
    parser.add_argument('-step', dest='step', type=int, default=16, help="gradient accumulation step")
    parser.add_argument('--backbone_freeze', dest='backbone_freeze', action='store_const', const=True, default=False, help='train the head only')
    # parser.add_argument('--uniform_loss', dest='weighted_loss', action='store_const', const=False, default=True, help='use the un-weighted loss for classes')
    parser.add_argument('--weighted_loss', dest='weighted_loss', action='store_const', const=True, default=False, help='use the weighted loss for classes')
    parser.add_argument('--restart', dest='restart', action='store_const', const=True, default=False, help='clear previous trained model')
    parser.add_argument('--debug', dest='debug', action='store_const', const=True, default=False, help='debug mode: small dataset')
    args = parser.parse_args()

    if args.model_name=='DeepLabV3':
        net = deeplabv3_resnet50(pretrained=True, aux_loss=False)
    else:
        raise NotImplementedError
    if args.backbone_freeze:
        net.requires_grad_(False)
        net.classifier = DeepLabHead(2048, 20)
        optimizer = optim.Adam(
            net.classifier.parameters(),
            lr = args.lr,
            weight_decay = 1e-4,
        )
    else:
        net.classifier = DeepLabHead(2048, 20)
        optimizer = optim.Adam(
            net.parameters(),
            lr = args.lr,
            weight_decay = 1e-4,
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch,eta_min=5e-7)
    model = ISSModelWrapper(
        net,
        task_criterion = nn.CrossEntropyLoss(weight=CITYSCAPES_CLASS_WEIGHT) if args.weighted_loss else nn.CrossEntropyLoss(),
        model_name = args.model_name,
        model_version = "fold-%1d"%args.k,
        model_root = "models/",
        optimizer = optimizer,
        scheduler = scheduler,
        tracker = [
            ('epoch','train_loss','less'),
            ('epoch','valid_loss','less'),
            ('epoch','train_mIoU','greater'),
            ('epoch','valid_mIoU','greater'),
        ],
        primary = 'valid_mIoU',
        device = 'cuda',
        clear = args.restart,
    )
    loaders = CityscapesLoaders(
        mode = 'fine',
        batch_sizes = {
            'train': 3,
            'extra': 3,
            'valid': 8,
            'testi': 8,
        },
        num_workers = 9,
        cross_validation = args.K,
        fold = args.k,
        debug = args.debug,
    )

    model.fit(loaders, num_epochs=args.epoch)