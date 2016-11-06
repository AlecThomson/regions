# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import math

import numpy as np
from astropy.coordinates import Angle
from astropy.wcs.utils import pixel_to_skycoord

from ..core import PixelRegion, SkyRegion
from ..utils import skycoord_to_pixel_scale_angle
from ..utils.positions_to_extents import get_phot_extents
from ..utils.wcs_helpers import skycoord_to_pixel_scale_angle
from .._geometry import circular_overlap_grid

__all__ = ['CirclePixelRegion', 'CircleSkyRegion']


class CirclePixelRegion(PixelRegion):
    """
    A circle in pixel coordinates.

    Parameters
    ----------
    center : `~regions.PixCoord`
        Center position
    radius : float
        Radius
    """

    def __init__(self, center, radius, meta=None, visual=None):
        # TODO: test that center is a 0D PixCoord
        self.center = center
        self.radius = radius
        self.meta = meta or {}
        self.visual = visual or {}

    def __repr__(self):
        data = dict(
            name=self.__class__.__name__,
            center=self.center,
            radius=self.radius,
        )
        fmt = '{name}\ncenter: {center}\nradius: {radius}'
        return fmt.format(**data)

    @property
    def area(self):
        return math.pi * self.radius ** 2

    def contains(self, pixcoord):
        return self.center.separation(pixcoord) < self.radius

    def to_shapely(self):
        return self.center.to_shapely().buffer(self.radius)

    def to_sky(self, wcs, mode='local', tolerance=None):
        if mode != 'local':
            raise NotImplementedError
        if tolerance is not None:
            raise NotImplementedError

        center = pixel_to_skycoord(self.center.x, self.center.y, wcs)
        # TODO: this is just called to compute `scale`
        # This is inefficient ... we should have that as a separate function.
        _, scale, _ = skycoord_to_pixel_scale_angle(center, wcs)

        radius = Angle(self.radius / scale, 'deg')
        return CircleSkyRegion(center, radius)

    def to_mask(self, shape, mode='center'):

        # For now we assume that this class represents a single circle

        # Find exact bounds
        xmin = self.center.x - self.radius
        xmax = self.center.x + self.radius
        ymin = self.center.y - self.radius
        ymax = self.center.y + self.radius

        # Find range of pixels
        imin = max(0, int(xmin))
        imax = min(shape[1] - 1, int(xmax) + 1)
        jmin = max(0, int(ymin))
        jmax = min(shape[0] - 1, int(ymax) + 1)

        # Find mask size
        nx = imax - imin + 1
        ny = jmax - jmin + 1

        # Find position of pixel edges and recenter so that circle is at origin
        xmin = float(imin) - 0.5 - self.center.x
        xmax = float(imax) + 0.5 - self.center.x
        ymin = float(jmin) - 0.5 - self.center.y
        ymax = float(jmax) + 0.5 - self.center.y

        if mode == 'center':
            use_exact = 0
        else:
            use_exact = 1

        print(xmin, xmax, ymin, ymax, nx, ny, self.radius, use_exact)

        fraction = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny,
                                         self.radius, use_exact, 1)

        if mode == 'all':
            return fraction == 1
        elif mode == 'any':
            return fraction > 0
        else:
            return fraction

    def as_patch(self, **kwargs):
        import matplotlib.patches as mpatches

        xy = self.center.x, self.center.y
        radius = self.radius
        patch = mpatches.Circle(xy=xy, radius=radius, **kwargs)
        return patch


class CircleSkyRegion(SkyRegion):
    """
    A circle in sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Center position
    radius : `~astropy.units.Quantity`
        Radius in angular units
    """

    def __init__(self, center, radius, meta=None, visual=None):
        # TODO: test that center is a 0D SkyCoord
        self.center = center
        self.radius = radius
        self.meta = meta or {}
        self.visual = visual or {}

    def __repr__(self):
        data = dict(
            name=self.__class__.__name__,
            center=self.center,
            radius=self.radius,
        )
        fmt = '{name}\ncenter: {center}\nradius: {radius}'
        return fmt.format(**data)

    @property
    def area(self):
        return math.pi * self.radius ** 2

    def contains(self, skycoord):
        return self.center.separation(skycoord) < self.radius

    def to_pixel(self, wcs, mode='local', tolerance=None):
        """
        Given a WCS, convert the circle to a best-approximation circle in pixel
        dimensions.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            A world coordinate system
        mode : 'local' or not
            not implemented
        tolerance : None
            not implemented

        Returns
        -------
        CirclePixelRegion
        """

        if mode != 'local':
            raise NotImplementedError
        if tolerance is not None:
            raise NotImplementedError

        center, scale, _ = skycoord_to_pixel_scale_angle(self.center, wcs)
        radius = self.radius.to('deg').value * scale

        return CirclePixelRegion(center, radius)

    def as_patch(self, ax, **kwargs):
        import matplotlib.patches as mpatches

        val = self.center.icrs
        center = (val.ra.to('deg').value, val.dec.to('deg').value)

        temp = dict(transform=ax.get_transform('icrs'),
                    radius=self.radius.to('deg').value)
        kwargs.update(temp)
        for key, value in self.visual.items():
            kwargs.setdefault(key, value)
        patch = mpatches.Circle(center, **kwargs)

        return patch
