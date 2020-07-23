from cutpathtool import pathitem
from math import pi, sqrt
import numpy as np
import pytest

@pytest.fixture()
def line():
    return pathitem.Line([-1., 4.], [3., 7.])

class TestLine:
    def test_start(self, line):
        np.testing.assert_almost_equal(line.start, [-1., 4.])

    def test_end(self, line):
        np.testing.assert_almost_equal(line.end, [3., 7.])

    def test_length(self, line):
        assert line.length == pytest.approx(5.)

    def test_reverse(self, line):
        line = line.reverse()
        np.testing.assert_almost_equal(line.start, [3., 7.])
        np.testing.assert_almost_equal(line.end, [-1., 4.])
        assert line.length == pytest.approx(5.)

@pytest.fixture()
def arc():
    return pathitem.Arc([0,5], 2, -1/6*pi, 3/4*pi, True)

class TestArc:
    def test_start(self, arc):
        np.testing.assert_almost_equal(arc.start, [sqrt(3), 4])
        np.testing.assert_almost_equal(arc.rad_start, -1/6*pi)

    def test_end(self, arc):
        np.testing.assert_almost_equal(arc.end, [-sqrt(2), sqrt(2) + 5])
        np.testing.assert_almost_equal(arc.rad_end, 3/4*pi)

    def test_length(self, arc):
        assert arc.length == pytest.approx(11/6*pi)

    def test_ccw(self, arc):
        assert arc.ccw == True

    def test_reverse(self, arc):
        arc = arc.reverse()
        np.testing.assert_almost_equal(arc.start, [-sqrt(2), sqrt(2) + 5])
        np.testing.assert_almost_equal(arc.rad_start, 3/4*pi)
        np.testing.assert_almost_equal(arc.end, [sqrt(3), 4])
        np.testing.assert_almost_equal(arc.rad_end, -1/6*pi)
        assert arc.length == pytest.approx(11/6*pi)
        assert arc.ccw == False

@pytest.fixture()
def circle():
    return pathitem.Circle([-4,0], 3, True)

class TestCircle:
    def test_start(self, circle):
        assert np.all(np.isnan(circle.start))

    def test_end(self, circle):
        assert np.all(np.isnan(circle.end))

    def test_length(self, circle):
        assert circle.length == pytest.approx(6*pi)

    def test_ccw(self, circle):
        assert circle.ccw == True

    def test_reverse(self, circle):
        circle = circle.reverse()
        assert circle.ccw == False
        assert np.all(np.isnan(circle.start))
        assert np.all(np.isnan(circle.end))
        assert circle.length == pytest.approx(6*pi)
