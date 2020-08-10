from abc import ABC, abstractmethod
from math import cos, sin, pi, fmod
from copy import deepcopy
import numpy as np

from math import acos, ceil, isclose

class Path(ABC):
    @property
    def start(self):
        # return self._start
        return self._ends[:,0]

    @property
    def end(self):
        # return self._end
        return self._ends[:,1]

    @property
    def length(self):
        return self._length

    @property
    def closed(self):
        return self._closed

    @abstractmethod
    def reverse(self):
        pass

    @abstractmethod
    def generate(self):
        pass

class Line(Path):
    def __init__(self, start, end):
        self._ends = np.transpose(np.array((start, end)))
        self._length = np.linalg.norm(self.start - self.end)
        self._closed = False

    def __str__(self):
        return 'Line(' + str(self.start) + ', ' + str(self.end) + ')'

    def reverse(self):
        line = deepcopy(self)
        line._ends = np.flip(line._ends, axis=1)
        return line

    def generate(self):
        return self._ends

class Arc(Path):
    def __init__(self, center, radius, rad_start, rad_end, ccw):
        self._center = np.array(center)
        self._radius = radius
        self._rad_ends = np.array([rad_start, rad_end])
        self._closed = False
        self._ccw = ccw

        # set both start and end in [0;2pi] range
        self._rad_ends = np.remainder(self._rad_ends, 2*pi)
        if isclose(self._rad_ends[0], self._rad_ends[1]):
            raise Exception('Arc must not be a circle, use Circle type instead')

        # reorder ends to never cross [-2pi;2pi] borders
        ends_ccw = self._rad_ends[0] < self._rad_ends[1]
        if self._ccw and not ends_ccw:
            self._rad_ends[0] -= 2*pi
        elif not self._ccw and ends_ccw:
            self._rad_ends[1] -= 2*pi

        # compute ends cartesian position
        self._ends = (self._center.reshape(2,1) + self._radius *
            np.vstack((np.cos(self._rad_ends), np.sin(self._rad_ends))))

        # compute lengths
        self._rad_length = self._rad_ends[1] - self._rad_ends[0]
        if not self._ccw:
            self._rad_length = -self._rad_length
        self._length = self._radius * self._rad_length

    def __str__(self):
        return ('Arc(' + str(self._center) + ', ' + str(self._radius) + ', ' +
                str(self.rad_start) + ', ' + str(self.rad_end) +
                (' CCW)' if self._ccw else ' CW)'))

    @property
    def rad_start(self):
        return self._rad_ends[0]

    @property
    def rad_end(self):
        return self._rad_ends[1]

    @property
    def ccw(self):
        return self._ccw

    def reverse(self):
        arc = deepcopy(self)
        arc._ends = np.flip(arc._ends, axis=1)
        arc._rad_ends = np.flip(arc._rad_ends)
        arc._ccw = not arc._ccw
        return arc

    def generate(self):
        max_error = 1e-2
        max_angle = 2 * acos(1.0 - max_error / self._radius)
        nb_segments = max(2, ceil(self._rad_length / max_angle) + 1)
        theta = np.linspace(self.rad_start, self.rad_end, nb_segments + 1)
        return (self._center.reshape(2,1) + self._radius *
                np.vstack((np.cos(theta), np.sin(theta))))

class Circle(Path):
    def __init__(self, center, radius, ccw):
        self._center = np.array(center)
        self._radius = radius
        self._ccw = ccw
        self._ends = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self._length = 2 * pi * self._radius
        self._closed = True

    def __str__(self):
        return ('Circle(' + str(self._center) + ', ' + str(self._radius) +
                (' CCW)' if self._ccw else ' CW)'))

    @property
    def ccw(self):
        return self._ccw

    def reverse(self):
        circle = deepcopy(self)
        circle._ccw = not circle._ccw
        return circle

    def generate(self):
        max_error = 1e-2
        max_angle = 2 * acos(1.0 - max_error / self._radius)
        nb_segments = max(2, ceil(2*pi / max_angle) + 1)
        if self._ccw:
            theta = np.linspace(0, 2*pi, nb_segments + 1)
        else:
            theta = np.linspace(2*pi, 0, nb_segments + 1)
        return (self._center.reshape(2,1) + self._radius *
                np.vstack((np.cos(theta), np.sin(theta))))

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
        self._ends = np.transpose(np.vstack((self._subpaths[0].start,
                                             self._subpaths[-1].end)))
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
        polypath._ends = np.flip(polypath._ends, axis=1)
        return polypath

    def generate(self):
        arrays = [p.generate()[:,:-1] for p in self._subpaths[:-1]]
        last_array = self._subpaths[-1].generate()
        arrays.append(last_array)
        return np.hstack(arrays)
