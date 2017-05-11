# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from ..core import PixCoord, PixelRegion, SkyRegion, BoundingBox

__all__ = ['PointPixelRegion', 'PointSkyRegion']


class PointPixelRegion(PixelRegion):
    """
    A point position in pixel coordinates.

    Parameters
    ----------
    center : `~regions.PixCoord`
        The position of the point
    """

    def __init__(self, center, meta=None, visual=None):
        # TODO: test that center is a 0D PixCoord
        self.center = center
        self.meta = meta or {}
        self.visual = visual or {}
        self._repr_params = None

    def contains(self, pixcoord):
        return False

    def to_shapely(self):
        return self.center.to_shapely()

    def to_sky(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError

    @property
    def bounding_box(self):
        # TODO: needs to be implemented
        raise NotImplementedError

    def to_mask(self, mode='center'):
        # TODO: needs to be implemented
        raise NotImplementedError

    def as_patch(self, **kwargs):
        # TODO: needs to be implemented
        raise NotImplementedError


class PointSkyRegion(SkyRegion):
    """
    A pixel region in sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the point
    """

    def __init__(self, center, meta=None, visual=None):
        # TODO: test that center is a 0D SkyCoord
        self.center = center
        self.meta = meta or {}
        self.visual = visual or {}
        self._repr_params = None

    def contains(self, skycoord):
        return False

    def to_pixel(self, wcs):
        """
        Given a WCS, return an PointPixelRegion which represents the same
        region but using pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            A world coordinate system

        Returns
        -------
        PointPixelRegion
        """
        center_x, center_y = skycoord_to_pixel(self.center, wcs=wcs)
        center = PixCoord(center_x, center_y)
        return PointPixelRegion(center)
