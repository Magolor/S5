export MASTER_ADDR=localhost
export MASTER_PORT=19260817
python -m torch.distributed.launch train_cityscapes.py --ddp --ddp_gpu=0,1,2,3,4,5,6,7 --restart $1