from cutpathtool import path
from math import pi, sqrt
import numpy as np
import pytest

@pytest.fixture()
def line():
    return path.Line([-1, 4], [3, 7])

class TestLine:
    def test_start(self, line):
        np.testing.assert_almost_equal(line.start, [-1, 4])

    def test_end(self, line):
        np.testing.assert_almost_equal(line.end, [3, 7])

    def test_length(self, line):
        assert line.length == pytest.approx(5.)

    def test_reverse(self, line):
        line = line.reverse()
        np.testing.assert_almost_equal(line.start, [3, 7])
        np.testing.assert_almost_equal(line.end, [-1, 4])
        assert line.length == pytest.approx(5)

    def test_closed(self, line):
        assert not line.closed

    def test_generate(self, line):
        assert np.allclose(line.generate(), np.array(([-1, 3], [4, 7])))

@pytest.fixture()
def arc():
    return path.Arc([0,5], 2, -1/6*pi, 3/4*pi, True)

class TestArc:
    def test_start(self, arc):
        np.testing.assert_almost_equal(arc.start, [sqrt(3), 4])
        np.testing.assert_almost_equal(arc.rad_start, -1/6*pi)

    def test_end(self, arc):
        np.testing.assert_almost_equal(arc.end, [-sqrt(2), sqrt(2) + 5])
        np.testing.assert_almost_equal(arc.rad_end, 3/4*pi)

    def test_length(self, arc):
        assert arc.length == pytest.approx(11/6*pi)

    def test_closed(self, arc):
        assert not arc.closed

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

    def test_generate(self, arc):
        result = arc.generate()
        assert np.allclose(result[:, 0], [sqrt(3), 4])
        assert np.allclose(result[:, -1], [-sqrt(2), sqrt(2) + 5])
        assert result.shape == (2, 17)

@pytest.fixture()
def circle():
    return path.Circle([-4,0], 3, True)

class TestCircle:
    def test_start(self, circle):
        assert np.all(np.isnan(circle.start))

    def test_end(self, circle):
        assert np.all(np.isnan(circle.end))

    def test_length(self, circle):
        assert circle.length == pytest.approx(6*pi)

    def test_closed(self, circle):
        assert circle.closed

    def test_ccw(self, circle):
        assert circle.ccw == True

    def test_reverse(self, circle):
        circle = circle.reverse()
        assert circle.ccw == False
        assert np.all(np.isnan(circle.start))
        assert np.all(np.isnan(circle.end))
        assert circle.length == pytest.approx(6*pi)

    def test_generate(self, circle):
        result = circle.generate()
        assert np.allclose(result[:,  0], [-1, 0])
        assert np.allclose(result[:, -1], [-1, 0])
        assert result.shape == (2, 41)

@pytest.fixture()
def open_polypath():
    return path.Polypath([path.Line([-2,-2], [0,0]),
                          path.Arc([2,0], 2, pi, 0, True),
                          path.Line([4,0], [4,2])])
@pytest.fixture()
def closed_polypath():
    return path.Polypath([path.Line([1,0], [3,2]),
                          path.Arc([3,1], 1, pi/2, -pi/2, False),
                          path.Line([3,0], [1,0])])

class TestPolypath:
    def test_start(self, open_polypath, closed_polypath):
        np.testing.assert_almost_equal(open_polypath.start, [-2, -2])
        np.testing.assert_almost_equal(closed_polypath.start, [1, 0])

    def test_end(self, open_polypath, closed_polypath):
        np.testing.assert_almost_equal(open_polypath.end, [4, 2])
        np.testing.assert_almost_equal(closed_polypath.end, [1, 0])

    def test_length(self, open_polypath, closed_polypath):
        assert open_polypath.length == pytest.approx(2*sqrt(2) + 2*pi + 2)
        assert closed_polypath.length == pytest.approx(2*sqrt(2) + pi + 2)

    def test_closed(self, open_polypath, closed_polypath):
        assert not open_polypath.closed
        assert closed_polypath.closed

    def test_reverse(self, open_polypath):
        open_polypath = open_polypath.reverse()
        np.testing.assert_almost_equal(open_polypath.start, [4, 2])
        np.testing.assert_almost_equal(open_polypath.end, [-2, -2])

    def test_generate(self, open_polypath, closed_polypath):
        result = open_polypath.generate()
        assert np.allclose(result[:,  0], [-2, -2])
        assert np.allclose(result[:, -1], [4, 2])
        assert result.shape == (2, 20)

        result = closed_polypath.generate()
        assert np.allclose(result[:,  0], [1, 0])
        assert np.allclose(result[:, -1], [1, 0])
        assert result.shape == (2, 16)
