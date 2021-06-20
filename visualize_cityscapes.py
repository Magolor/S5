from train_cityscapes import *
from PIL import Image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_backbone', dest='model_backbone', choices=['ResNet50','DenseNet121','ResNeSt50','HRNetW48'], default='HRNetW48', help="type of model to use")
    parser.add_argument('-cross', dest='K', type=int, default=0, help="cross validation's fold number (<= 0 for no cross validation)")
    parser.add_argument('-fold', dest='k', type=int, default=0, help="cross validation's fold index (0 <= index < cross_validation)")
    args = parser.parse_args()
    
    model = ModelLoader(
        model_root = "models/",
        model_name = "DeepLabV3-"+args.model_backbone+"-F",
        model_version = "fold-%1d"%args.k,
        file = 'e000170.pth',
        device = 'cuda',
    )
    loaders = CityscapesLoaders(
        mode = 'fine',
        batch_sizes = {
            'train': 2,
            'valid': 2,
            'testi': 2,
        },
        num_workers = 16,
        cross_validation = args.K,
        fold = args.k,
        ddp = False,
        debug = True,
    )

    print(model.epoch)
    stats, outputs = model.valid_epoch(loaders['valid'],return_valid_stats=True,return_outputs=True)
    stats['valid_mIoU'] = IandU_to_mIoU(stats['valid_I'], stats['valid_U'])
    print("Loss:",stats['valid_loss'])
    print("mIoU:",stats['valid_mIoU'])

    P = outputs.argmax(dim=1)
    F = [[np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)] for _ in range(35)]
    for i,p in enumerate(P):
        I = p.long().numpy()
        C = np.array([ [F[I[row][col]] for col in range(I.shape[1])] for row in range(I.shape[0])],dtype=np.uint8)
        Image.fromarray(C).save("visualize/%d.png"%i)