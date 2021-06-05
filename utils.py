import os
import re
import sys
import math
import time
import tqdm
import json
import torch
import shutil
import logging
import requests
import jsonlines
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== File Operations =====
def Folder(PATH):
    return "/".join(PATH.split('/')[:-1])+"/"
    
def File(PATH):
    return PATH.split('/')[-1]

def Prefix(PATH):
    return ".".join(PATH.split('.')[:-1])

def Suffix(PATH):
    return PATH.split('.')[-1]

def Create(PATH):
    None if os.path.exists(PATH) else os.makedirs(PATH)

def Delete(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None

def Clear(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None; os.makedirs(PATH)


def SaveJSON(object, FILE, jsonl=False, indent=None):
    if jsonl:
        with jsonlines.open(FILE, 'w') as f:
            for data in object:
                f.write(data)
    else:
        with open(FILE, 'w') as f:
            json.dump(object, f, indent=indent)

def PrettifyJSON(PATH):
    if PATH[-1]=='/':
        for FILE in os.listdir(PATH):
            SaveJSON(LoadJSON(PATH+FILE),PATH+FILE,indent=4)
    else:
        SaveJSON(LoadJSON(PATH),PATH,indent=4)

def LoadJSON(FILE, jsonl=False):
    if jsonl:
        with open(FILE, 'r') as f:
            return [data for data in jsonlines.Reader(f)]
    else:
        with open(FILE, 'r') as f:
            return json.load(f)

def View(something, length=4096):
    print(str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewS(something, length=4096):
    return (str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewDict(something, length=4096, limit=512):
    print("{")
    for i,item in enumerate(something.items()):
        print("\t"+str(item[0])+": "+(ViewS(item[1],length)+','))
        if i>=limit:
            print("\t..."); break
    print("}")

def ViewDictS(something, length=4096, limit=512):
    s = "{\n"
    for i,item in enumerate(something.items()):
        s += "\t"+str(item[0])+": "+(ViewS(item[1],length)+',')+"\n"
        if i>=limit:
            s += "\t...\n"; break
    s += "}\n"; return s

def ViewJSON(json_dict, length=4096):
    print(ViewS(json.dumps(json_dict,indent=4),length))

def ViewJSONS(json_dict, length=4096):
    return ViewS(json.dumps(json_dict,indent=4),length)

# ===== Helper Functions =====
def IP():
    return requests.get('https://api.ipify.org').text

def DATE():
    return time.strftime("%Y-%m-%d",time.localtime(time.time()))

def TQDM(something, s=0, desc=None):
    if type(something) is int:
        return tqdm.trange(s,something+s,desc=desc,dynamic_ncols=True)
    else:
        return zip(tqdm.trange(s,len(list(something))+s,desc=desc,dynamic_ncols=True),list(something))

def CMD(command, wait=True):
    h = subprocess.Popen(command,shell=True); return h.wait() if wait else h

def LineToFloats(line):
    return [float(s) for s in re.findall(r"(?<!\w)[-+]?\d*\.?\d+(?!\d)",line)]

def PrintConsole(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)

def PrintError(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def NORMAL(something):
    return str(something)

def ERROR(something):
    return "\033[1;31m"+str(something)+"\033[0m"

def SUCCESS(something):
    return "\033[1;32m"+str(something)+"\033[0m"

def WARN(something):
    return "\033[1;33m"+str(something)+"\033[0m"

def HIGHLIGHT(something):
    return "\033[1;34m"+str(something)+"\033[0m"

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True
def Logger(PATH, level='debug', console=False, mode='a'):
    _level = logging.DEBUG if level=='debug' else logging.INFO
    logger = logging.getLogger(); logger.setLevel(_level)
    if console:
        cs = logging.StreamHandler(); cs.setLevel(_level); logger.addHandler(cs)
    if PATH is not None and PATH != '':
        Create(Folder(PATH)); fh = logging.FileHandler(PATH, mode=mode); fh.setLevel(_level); logger.addHandler(fh)
    return logger

# ===== With Clause =====
class Timer():
    def __init__(self, NAME="Timer", COLOR_SCHEME=NORMAL):
        self.name = NAME; self.COLOR = COLOR_SCHEME
    def __enter__(self):
        self.start_time = time.time(); self.int_start_time = int(self.start_time)
        print(self.COLOR("[%s Start]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(self.int_start_time)))); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time(); self.int_end_time = int(self.end_time); interval = self.end_time - self.start_time
        print(self.COLOR("[%s   End]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(  self.int_end_time))))
        print(self.COLOR("[%s Total]: {:02d}h.{:02d}m.{:02d}s.{:03d}ms.".format(int(interval)//3600,int(interval)%3600//60,int(interval)%60,int(interval*1000)-int(interval)*1000)%self.name))
    def tick(self, desc=None):
        cur_time = time.time(); int_cur_time = int(cur_time); interval = cur_time - self.start_time
        print(self.COLOR("[%s  Tick]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(       cur_time))),self.COLOR("" if desc is None else "(%s)"%desc))
        print(self.COLOR("[%s Sofar]: {:02d}h.{:02d}m.{:02d}s.{:03d}ms.".format(int(interval)//3600,int(interval)%3600//60,int(interval)%60,int(interval*1000)-int(interval)*1000)%self.name))

class Painter():
    def __init__(self, title, FILE, figsize=(16,9)):
        self.title = title; self.FILE = FILE; self.figsize=figsize
    def __enter__(self):
        fig,axe = plt.subplots(figsize=self.figsize,dpi=300); axe.set_title(self.title); self.axe=axe; return (fig,axe) 
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.savefig(self.FILE); plt.close()

class Tracker():
    def __init__(self, title, DIR, registrations=[]):
        self.title = title; self.DIR=DIR; Create(DIR); self.curve = {}; self.xlabel = {}; self.key = {}
        self.data_file = os.path.join(self.DIR, "%s.dat"%self.title)
        for key in registrations:
            X, Y, b = key; self.curve[Y] = []; self.xlabel[Y] = X; self.key[Y] = b
    def compare_func(self, b):
        if b=='greater':
            return lambda x: x
        if b=='less':
            return lambda x: -x
        if b=='none':
            return None
        return b
    def variable_profile(self, variable):
        values = self.curve[variable]
        key = self.compare_func(self.key[variable])
        return values[np.argmax([key(x[1]) for x in values])][1]
    def profile(self):
        profile = {}
        for variable in self.curve.keys():
            profile[variable] = self.variable_profile(variable)
        return profile
    def update(self, variable, value, time):
        if value is None:
            return False
        assert(variable in self.curve)
        self.curve[variable].append((time,value)); key = self.compare_func(self.key[variable])
        return False if key is None else (key(value)>=max([key(x[1]) for x in self.curve[variable]]))
    def load(self):
        if os.path.exists(self.data_file):
            self.__dict__.update(dict(torch.load(self.data_file)))
    def plot(self):
        for variable, values in self.curve.items():
            if self.key[variable]!='none':
                values = sorted(values); X,Y = [v[0] for v in values], [v[1] for v in values]
                with Painter("%s: %s"%(self.title,variable), os.path.join(self.DIR,"%s.png"%variable)) as (fig,axe):
                    plt.plot(X, Y); plt.xlabel(self.xlabel[variable]); plt.ylabel(variable)
    def save(self):
        torch.save(self.__dict__, self.data_file); self.plot()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

def LoadTracker(title, DIR):
    T = Tracker(title,DIR); T.load(); T.save(); return T

def MeanTracker(title, DIR, trackers):
    T = Tracker(title,DIR)
    if len(trackers)==0:
        T.save(); return T
    else:
        T.xlabel = dict(trackers[0].xlabel); T.key = dict(trackers[0].key)
        T.curve = {variable:[] for variable in trackers[0].curve.keys()}
        for variable in T.curve.keys():
            for i in range(min([len(tracker.curve[variable]) for tracker in trackers])):
                T.curve[variable].append((trackers[0].curve[variable][i][0],float(np.mean([
                    tracker.curve[variable][i][1] for tracker in trackers
                ]))))
        T.save(); return T
    
