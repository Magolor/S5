from transforms_vision import *
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

def CityscapesLoaders(
    mode = 'fine',
    transforms = {
        'train': AUGMENT_TRANSFORM,
        'valid': DEFAULT_TRANSFORM,
        'testi': DEFAULT_TRANSFORM,
    },
    batch_sizes = {
        'train': 3,
        'valid': 3,
        'testi': 3,
    },
    num_workers = 9,
    cross_validation = -1,
    fold = 0,
    ddp = False,
    seed = 19260817,
    return_datasets = False,
    debug = False,
):
    datasets = {
        'train': Cityscapes('./data/', split='train', mode=mode, target_type='semantic', transforms=transforms['train']),
        'valid': Cityscapes('./data/', split='val', mode=mode, target_type='semantic', transforms=transforms['valid']),
    }

    if cross_validation > 0:
        fold_length = len(datasets['train'])//cross_validation
        indices = [i for i in range(len(datasets['train']))]; np.random.seed(seed); np.random.shuffle(indices)
        train_valid = Cityscapes('./data/', split='train', mode=mode, target_type='semantic', transforms=transforms['valid'])
        train_mask = [i for i in range(len(datasets['train'])) if i not in indices[fold*fold_length:(fold+1)*fold_length]]
        valid_mask = [i for i in range(len(datasets['train'])) if i     in indices[fold*fold_length:(fold+1)*fold_length]]
        new_train = Subset(datasets['train'], train_mask); new_valid = Subset(train_valid, valid_mask)
        datasets['testi'] = datasets['valid']; datasets['valid'] = new_valid; datasets['train'] = new_train
    else:
        datasets['testi'] = datasets['valid']

    if debug:
        SAMPLE_LENGTH = 64
        indices = {
            'train': np.random.choice(len(datasets['train']),SAMPLE_LENGTH),
            'valid': np.random.choice(len(datasets['valid']),SAMPLE_LENGTH),
            'testi': np.random.choice(len(datasets['testi']),SAMPLE_LENGTH),
        }
        datasets = {
            'train': Subset(datasets['train'], indices['train']),
            'valid': Subset(datasets['valid'], indices['valid']),
            'testi': Subset(datasets['testi'], indices['testi']),
        }

    if ddp:
        samplers = {
            'train': DistributedSampler(datasets['train'], shuffle=True),
            'valid': DistributedSampler(datasets['valid']),
            'testi': DistributedSampler(datasets['testi']),
        }
    else:
        samplers = {
            'train': None,
            'valid': None,
            'testi': None,
        }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_sizes['train'], sampler=samplers['train'], shuffle=not ddp, num_workers=num_workers, pin_memory=False, drop_last=True),
        'valid': DataLoader(datasets['valid'], batch_size=batch_sizes['valid'], sampler=samplers['valid'], shuffle=  False, num_workers=num_workers, pin_memory=False, drop_last=True),
        'testi': DataLoader(datasets['testi'], batch_size=batch_sizes['testi'], sampler=samplers['testi'], shuffle=  False, num_workers=num_workers, pin_memory=False, drop_last=True),
    }

    return (dataloaders, datasets) if return_datasets else dataloaders

if __name__=="__main__":
    loaders = CityscapesLoaders(cross_validation=5, fold=1)
    for data in loaders['train']:
        print(data[0].shape, data[1].shape); break
    for data in loaders['valid']:
        print(data[0].shape, data[1].shape); exit(0)