from pickle import load
from utils import *
import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP

class ModelWrapper(nn.Module):
    def __init__(self,
        net,
        model_root = "models/",
        model_name = "DefaultNet",
        model_version = "v1",
        optimizer = None,
        scheduler = None,
        console_log = True,
        tracker = [('epoch','train_acc','greater'), ('epoch','train_loss','less'), ('epoch','valid_acc','greater'), ('epoch','valid_loss','less')],
        primary = 'valid_acc',
        epoch = 0,
        ddp = False,
        device = 'cpu',
        clear = False,
    ):
        super(ModelWrapper, self).__init__()
        self.model_root = model_root; self.model_name = model_name; self.model_version = model_version
        self.net = net; self.path = os.path.join(model_root, model_name, model_version); Clear(self.path) if clear else None
        self.logger = Logger(os.path.join(self.path,"stats/log.log"),console=console_log,mode='w' if clear else 'a')
        self.tracker = Tracker(model_name+"-"+model_version,os.path.join(self.path,"stats"),tracker); self.tracker.load()
        self.optimizer = optimizer; self.scheduler = scheduler; self.console_log = console_log; self.primary = primary
        self.epoch = epoch; self.ddp = ddp; self.to(device)
        if ddp:
            self.net = DDP(self.net,find_unused_parameters=True)
        self.save()

    def to(self, device):
        self.device = device; self.net.to(device); return self

    def eval(self):
        self.net.eval(); return self

    def train(self):
        self.net.train(); return self

    def save(self, file=None):
        file = "e%06d.pth"%self.epoch if file is None else file; FILE = os.path.join(self.path,file)
        tmp_logger = self.logger; self.logger = None
        self.tracker.save(); self.tracker = None
        if self.ddp:
            tmp_ddp = self.net; self.net = self.net.module
        torch.save(self,FILE);
        if self.ddp:
            self.net = tmp_ddp
        self.tracker = LoadTracker(self.model_name+"-"+self.model_version,os.path.join(self.path,"stats"))
        self.logger = tmp_logger
        return FILE
    
    def load(self, file):
        if os.path.exists(os.path.join(self.path,file)):
            self.__dict__.update(dict(vars(torch.load(os.path.join(self.path,file)))))
            self.logger = Logger(os.path.join(self.path,"log.log"),console=self.console_log,mode='a')
        self.to(self.device); return self

    def clone(self, device):
        model = torch.load(self.save()); model.load(); return model.to(device)

    def clear_log(self):
        self.logger = Logger(os.path.join(self.path,"log.log"),console=self.console_log,mode='w')

    # Change for different models or different tasks
    def forward(self, x):
        return self.net(x.to(self.device))

    # Change for different models or different tasks
    def run_batch(self, data, return_stats=False, return_outputs=False):
        raise NotImplementedError

    # Change for different models or different tasks
    def train_epoch(self, train_loader, gradient_step=1, return_train_stats=False, return_outputs=False):
        self.train(); loss = 0.; outputs = []; self.epoch += 1
        if hasattr(train_loader.sampler,"set_epoch"):
            train_loader.sampler.set_epoch(self.epoch)
        with torch.enable_grad():
            train_stats = defaultdict(float); c = 0;
            pbar = tqdm.tqdm(total=len(train_loader),unit='batch')
            for batch, data in enumerate(train_loader):
                c += 1; result = self.run_batch(data, return_stats= return_train_stats, return_outputs=return_outputs); loss += result['loss'].item()
                result['loss'].backward()
                if (batch+1)%gradient_step==0 or batch+1==len(train_loader):
                    self.optimizer.step(); self.optimizer.zero_grad()
                if result['stats'] is not None:
                    for key, value in result['stats'].items():
                        train_stats["train_"+key] += value
                if result['outputs'] is not None:
                    outputs.append(result['outputs'])
                pbar.set_description(SUCCESS("[%s]"%self.model_name)+" Epoch %s Train (loss=%7.4f)"%(HIGHLIGHT("#%04d"%self.epoch),result['loss'].item())); pbar.update()
                if self.console_log:
                    self.logger.info("[%s] Epoch #%04d Batch #%04d/%04d Train Loss = %7.4f"%(self.model_name,self.epoch,batch+1,len(train_loader),result['loss'].item()))
            pbar.close()
            for key in train_stats.keys():
                train_stats[key] /= c
        if self.console_log:
            self.logger.info("[%s] Epoch #%04d Train Loss = %7.4f"%(self.model_name,self.epoch,loss/c))
        if self.scheduler is not None:
            self.scheduler.step()
        if return_outputs:
            outputs = torch.cat(outputs,dim=0)
        return (
            (train_stats,outputs) if (return_train_stats and return_outputs) else (
                train_stats if return_train_stats else (
                    outputs if return_train_stats else None
                )
            )
        )

    # Change for different models or different tasks
    def valid_epoch(self, valid_loader, return_valid_stats= True, return_outputs=False, tab_desc=False):
        self.eval(); loss = 0.; outputs = []
        with torch.no_grad():
            valid_stats = defaultdict(float); c = 0
            pbar = tqdm.tqdm(total=len(valid_loader),unit='batch')
            for data in valid_loader:
                c += 1; result = self.run_batch(data, return_stats= return_valid_stats, return_outputs=return_outputs); loss += result['loss'].item()
                if result['stats'] is not None:
                    for key, value in result['stats'].items():
                        valid_stats["valid_"+key] += value
                if result['outputs'] is not None:
                    outputs.append(result['outputs'])
                if tab_desc:
                    pbar.set_description(" "*len("[%s]"%self.model_name+" Epoch #%04d "%self.epoch)+"Valid (loss=%7.4f)"%(result['loss'].item())); pbar.update()
                else:
                    pbar.set_description(SUCCESS("[%s]"%self.model_name)+" Epoch %s Valid (loss=%7.4f)"%(HIGHLIGHT("#%04d"%self.epoch),result['loss'].item())); pbar.update()
            pbar.close()
            for key in valid_stats.keys():
                valid_stats[key] /= c
        if self.console_log:
            self.logger.info("[%s] Epoch #%04d Valid Loss = %7.4f"%(self.model_name,self.epoch,loss/c))
        if return_outputs:
            outputs = torch.cat(outputs,dim=0)
        return (
            (valid_stats,outputs) if (return_valid_stats and return_outputs) else (
                valid_stats if return_valid_stats else (
                    outputs if return_outputs else None
                )
            )
        )

    # Change for different models or different tasks
    def run_epoch(self, loaders, gradient_step=True, validate_per=1, save_per=None, return_train_stats=True, return_valid_stats=True):
        if gradient_step>0:
            train_stats = self.train_epoch(loaders['train'], gradient_step=gradient_step, return_train_stats=return_train_stats)
        if validate_per is not None and self.epoch%validate_per==0:
            valid_stats = self.valid_epoch(loaders['valid'], return_valid_stats=return_valid_stats, tab_desc=True)
        self.tracker.save()
        if save_per is not None and self.epoch%save_per==0:
            self.save()
        return {
            'train_stats': train_stats if return_train_stats else None,
            'valid_stats': valid_stats if return_valid_stats else None,
        }

    # Change for different models or different tasks
    def fit(self, loaders, num_epochs=100, validate_per=1, save_per=None):
        for _ in range(num_epochs):
            result = self.run_epoch(loaders, training=True, validate_per=validate_per, save_per=save_per)
            for key, value in result['train_stats'].items():
                _ = self.tracker.update(key, value, self.epoch)
            for key, value in result['valid_stats'].items():
                u = self.tracker.update(key, value, self.epoch)
                if u and key==self.primary:
                    self.save("best.pth")

def ModelLoader(model_root, model_name, model_version, file, device = 'cpu'):
    model = torch.load(os.path.join(model_root, model_name, model_version, file), map_location=device); return model.to(device)