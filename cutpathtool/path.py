from abc import ABC, abstractmethod
from math import cos, sin, pi, fmod
from copy import deepcopy
import numpy as np

class Path(ABC):
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def length(self):
        return self._length

    @property
    def closed(self):
        return self._closed

    @abstractmethod
    def reverse(self):
        pass

class Line(Path):
    def __init__(self, start, end):
        self._start = np.array(start)
        self._end = np.array(end)
        self._length = np.linalg.norm(self._start - self._end)
        self._closed = False

    def __str__(self):
        return 'Line(' + str(self._start) + ', ' + str(self._end) + ')'

    def reverse(self):
        line = deepcopy(self)
        line._start, line._end = self._end, self._start
        return line

class Arc(Path):
    def __init__(self, center, radius, rad_start, rad_end, ccw):
        self._center = np.array(center)
        self._radius = radius
        self._rad_start = rad_start
        self._rad_end = rad_end
        self._ccw = ccw
        self._closed = False

        self._start = (self._center + self._radius *
                       np.array([cos(self._rad_start), sin(self._rad_start)]))
        self._end = (self._center + self._radius *
                     np.array([cos(self._rad_end), sin(self._rad_end)]))

        self._rad_len = self._rad_end - self._rad_start
        if not self._ccw:
            self._rad_len = -self._rad_len
        self._rad_len = fmod(self._rad_len + 2*pi, 2*pi)

        self._length = self._radius * self._rad_len

    def __str__(self):
        return ('Arc(' + str(self._center) + ', ' + str(self._radius) + ', ' +
                str(self._rad_start) + ', ' + str(self._rad_end) + ')')

    @property
    def rad_start(self):
        return self._rad_start

    @property
    def rad_end(self):
        return self._rad_end

    @property
    def ccw(self):
        return self._ccw

    def reverse(self):
        arc = deepcopy(self)
        arc._start, arc._end = self._end, self._start
        arc._rad_start, arc._rad_end = (self._rad_end, self._rad_start)
        arc._ccw = not arc._ccw
        return arc

class Circle(Path):
    def __init__(self, center, radius, ccw):
        self._center = np.array(center)
        self._radius = radius
        self._ccw = ccw
        self._start = self._end = np.array([np.nan, np.nan])
        self._length = 2 * pi * self._radius
        self._closed = True

    def __str__(self):
        return 'Circle(' + str(self._center) + ', ' + str(self._radius) + ')'

    @property
    def ccw(self):
        return self._ccw

    def reverse(self):
        circle = deepcopy(self)
        circle._ccw = not circle._ccw
        return circle

class Polypath(Path):
    # no verification made on path connectivity
    def __init__(self, paths):
        if not paths:
            raise Exception('Empty path list')
        self._length = 0.
        self._subpaths = []
        for p in paths:
            self._length += p.length
            if isinstance(p, Polypath):
                self._subpaths += p._subpaths
            else:
                self._subpaths.append(p)
        self._start = self._subpaths[0].start
        self._end = self._subpaths[-1].end
        self._closed = np.allclose(self.start, self.end)

    def __str__(self):
        s = 'Polypath(' + str(self._subpaths[0])
        for p in self._subpaths[1:]:
            s += ', ' + str(p)
        return s + ')'

    def reverse(self):
        subpaths_save = self._subpaths
        self._subpaths = [] # remove subpaths
        polypath = deepcopy(self) # copy with empty subpaths
        self._subpaths = subpaths_save # restore subpaths
        polypath._subpaths = [p.reverse() for p in reversed(self._subpaths)]
        polypath._start, polypath._end = polypath._end, polypath._start
        return polypath
