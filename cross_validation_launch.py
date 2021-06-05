from utils import *

MODEL = "DeepLabV3"
COMMAND = "CUDA_VISIBLE_DEVICES={0} python3 train_cityscapes.py -cross {1} -fold {0} -model_name {2} --restart"
KFOLD = 6
TRAIN = True

if __name__=="__main__":
    if TRAIN:
        handles = [CMD(COMMAND.format(fold,KFOLD,MODEL),wait=False) for fold in range(KFOLD)]
        finished_handles = [h.wait() for h in handles];
    STATS = "stats/" + MODEL + "/"; Clear(STATS); trackers = []
    for fold, directory in enumerate(sorted(os.listdir("models/" + MODEL + "/"))):
        SRC = "models/" + MODEL + "/" + directory + "/stats"; DST = STATS + directory
        CMD("cp -r {0} {1}".format(SRC, DST)); trackers.append(LoadTracker(title=MODEL+'-'+directory,DIR=DST))
    T = MeanTracker(title=MODEL,DIR=STATS+"summary",trackers=trackers)
