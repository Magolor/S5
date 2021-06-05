from transforms_vision import *
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, Subset

def CityscapesLoaders(
    mode = 'fine',
    transforms = {
        'train': AUGMENT_TRANSFORM,
        'extra': AUGMENT_TRANSFORM,
        'valid': DEFAULT_TRANSFORM,
        'testi': DEFAULT_TRANSFORM,
    },
    batch_sizes = {
        'train': 3,
        'extra': 3,
        'valid': 3,
        'testi': 3,
    },
    num_workers = 9,
    cross_validation = -1,
    fold = 0,
    seed = 19260817,
    return_datasets = False,
    debug = False,
):
    datasets = {
        'train': Cityscapes('./data/', split='train', mode=mode, target_type='semantic', transforms=transforms['train']),
        'extra': Cityscapes('./data/', split='train_extra', mode=mode, target_type='semantic', transforms=transforms['extra']) if mode=='coarse' else None,
        'valid': Cityscapes('./data/', split='val', mode=mode, target_type='semantic', transforms=transforms['valid']),
        'testi': Cityscapes('./data/', split='test', mode=mode, target_type='semantic', transforms=transforms['testi']),
    }

    if cross_validation > 0:
        fold_length = len(datasets['train'])//cross_validation
        indices = [i for i in range(len(datasets['train']))]; np.random.seed(seed); np.random.shuffle(indices)
        train_valid = Cityscapes('./data/', split='train', mode=mode, target_type='semantic', transforms=transforms['valid'])
        train_mask = [i for i in range(len(datasets['train'])) if i not in indices[fold*fold_length:(fold+1)*fold_length]]
        valid_mask = [i for i in range(len(datasets['train'])) if i     in indices[fold*fold_length:(fold+1)*fold_length]]
        new_train = Subset(datasets['train'], train_mask); new_valid = Subset(train_valid, valid_mask)
        datasets['testi'] = datasets['valid']; datasets['valid'] = new_valid; datasets['train'] = new_train

    if not debug:
        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_sizes['train'], shuffle= True, num_workers=num_workers, pin_memory=True, drop_last=True),
            'extra': DataLoader(datasets['extra'], batch_size=batch_sizes['extra'], shuffle= True, num_workers=num_workers, pin_memory=True, drop_last=True) if mode=='coarse' else None,
            'valid': DataLoader(datasets['valid'], batch_size=batch_sizes['valid'], shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True),
            'testi': DataLoader(datasets['testi'], batch_size=batch_sizes['testi'], shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True),
        }
    else:
        SAMPLE_LENGTH = 48
        indices = {
            'train': [i for i in range(SAMPLE_LENGTH)],
            'extra': [i for i in range(SAMPLE_LENGTH)] if mode=='coarse' else None,
            'valid': [i for i in range(SAMPLE_LENGTH)],
            'testi': [i for i in range(SAMPLE_LENGTH)],
        }
        dataloaders = {
            'train': DataLoader(Subset(datasets['train'], indices['train']), batch_size=batch_sizes['train'], shuffle= True, num_workers=num_workers, pin_memory=True, drop_last=True),
            'extra': DataLoader(Subset(datasets['extra'], indices['extra']), batch_size=batch_sizes['extra'], shuffle= True, num_workers=num_workers, pin_memory=True, drop_last=True) if mode=='coarse' else None,
            'valid': DataLoader(Subset(datasets['valid'], indices['valid']), batch_size=batch_sizes['valid'], shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True),
            'testi': DataLoader(Subset(datasets['testi'], indices['testi']), batch_size=batch_sizes['testi'], shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True),
        }

    return (dataloaders, datasets) if return_datasets else dataloaders

if __name__=="__main__":
    loaders = CityscapesLoaders(cross_validation=5, fold=1)
    for data in loaders['train']:
        print(data[0].shape, data[1].shape); break
    for data in loaders['valid']:
        print(data[0].shape, data[1].shape); exit(0)