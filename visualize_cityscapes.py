from train_cityscapes import *
from PIL import Image

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
    parser.add_argument('--suggestion', dest='suggestion', type=str, default="500", help='suggestion defined by \'box\', \'click\', or an integer')
    parser.add_argument('--exp', default="", type=str, help='experiment name')
    parser.add_argument('--ddp_gpu', default="0", type=str, help='gpu indices')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()
    
    model = ModelLoader(
        model_root = "models/",
        model_name = "DeepLabV3-"+args.model_backbone+"-"+args.exp,
        model_version = "fold-%1d"%args.k,
        file = 'best.pth',
        device = 'cuda',
    )
    TRANSFORM = CREATE_SUGGEST_TRANSFORM(suggestion = args.suggestion)
    loaders, dataset = CityscapesLoaders(
        mode = 'fine',
        transforms = {
            'train': TRANSFORM,
            'valid': TRANSFORM,
            'testi': TRANSFORM,
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
        debug = True,
        return_datasets = True,
    )

    print(model.epoch)
    stats, outputs = model.valid_epoch(loaders['valid'],return_valid_stats=True,return_outputs=True)
    stats['valid_pd_mIoU'] = IandU_to_mIoU(stats['valid_pd_I'], stats['valid_pd_U'])
    stats['valid_sg_mIoU'] = IandU_to_mIoU(stats['valid_sg_I'], stats['valid_sg_U'])
    stats['valid_bs_mIoU'] = stats['valid_pd_mIoU'] - stats['valid_sg_mIoU']
    print("Loss:",stats['valid_loss'])
    print("pd mIoU:",stats['valid_pd_mIoU'])
    print("sg mIoU:",stats['valid_sg_mIoU'])
    print("bs mIoU:",stats['valid_bs_mIoU'])

    P = outputs.argmax(dim=1)
    F = [[np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)] for _ in range(35)]
    dataset['valid'].dataset.transforms = None
    for i,p in enumerate(P):
        I = p.long().numpy()
        C = np.array([ [F[I[row][col]] for col in range(I.shape[1])] for row in range(I.shape[0])],dtype=np.uint8)
        Image.fromarray(C).save("visualize/%d.png"%i)
        dataset['valid'][i][0].save("visualize/%d.orig.png"%i)