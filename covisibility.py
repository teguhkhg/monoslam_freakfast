from threading import Lock

from collections import defaultdict, Counter
from itertools import chain


class GraphMapPoint(object):
    def __init__(self):
        self.id = None
        self.meas = dict()
        self._lock = Lock()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMapPoint) and 
            self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id
        
    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):
        with self._lock:
            return self.meas.keys()

    def keyframes(self):
        with self._lock:
            return self.meas.values()

    def add_measurement(self, m):
        with self._lock:
            self.meas[m] = m.keyframe

    def remove_measurement(self, m):
        with self._lock:
            try:
                del self.meas[m]
            except KeyError:
                pass


class GraphMeasurement(object):
    def __init__(self):
        self.keyframe = None
        self.mappoint = None

    @property
    def id(self):
        return (self.keyframe.id, self.mappoint.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMeasurement) and
            self.id == rhs.id)