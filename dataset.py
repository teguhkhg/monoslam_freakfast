import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue

class ImageReader(object):
    def __init__(self, ids, timestamps, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10
        self.waiting = 1.5

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)

    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue

            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]
        
    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape

class KITTIOdometry(object):
    def __init__(self, path):
        Cam = namedtuple("cam", "fx fy cx cy width height")
        cam00_02 = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376)
        cam03 = Cam(721.5377, 721.5377, 609.5593, 172.854, 1241, 376)
        cam04_12 = Cam(707.0912, 707.0912, 601.8873, 183.1104, 1241, 376)

        os.path.expanduser(path)
        self.timestamps = np.loadtxt(os.path.join(path, "times.txt"))
        self.reader = ImageReader(self.listdir(os.path.join(path, "image_0")), self.timestamps)

        sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])
        if sequence < 3:
            self.cam = cam00_02
        elif sequence == 3:
            self.cam = cam03
        elif sequence < 13:
            self.cam = cam04_12

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))
    
    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.reader)

