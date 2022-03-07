# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module provides several custom descriptor classes for attribute
validation of region classes.
"""

import abc

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import numpy as np

from .pixcoord import PixCoord

__all__ = []


class RegionAttribute(abc.ABC):
    """
    Base descriptor class for region attribute validation.

    Parameters
    ----------
    description : str, optional
        The description of the attribute, which will be used as the
        attribute documentation.
    """

    def __init__(self, description=''):
        self.__doc__ = description

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self  # pragma: no cover
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        self._validate(value)
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError(f'cannot delete {self.name!r}')

    @abc.abstractmethod
    def _validate(self, value):
        """
        Validate the attribute value.

        An exception is raised if the value is invalid.
        """
        raise NotImplementedError  # pragma: no cover


class ScalarPixCoord(RegionAttribute):
    """
    Descriptor class to check that value is a scalar `~regions.PixCoord`.
    """

    def _validate(self, value):
        if not (isinstance(value, PixCoord) and value.isscalar):
            raise ValueError(f'{self.name!r} must be a scalar PixCoord')


class OneDPixCoord(RegionAttribute):
    """
    Descriptor class to check that value is a 1D `~regions.PixCoord`.
    """

    def _validate(self, value):
        if not (isinstance(value, PixCoord) and not value.isscalar
                and value.x.ndim == 1):
            raise ValueError(f'{self.name!r} must be a 1D PixCoord')


class PositiveScalar(RegionAttribute):
    """
    Descriptor class to check that value is a strictly positive (> 0)
    scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value) or value <= 0:
            raise ValueError(f'{self.name!r} must be a positive scalar')


class ScalarSkyCoord(RegionAttribute):
    """
    Descriptor class to check that value is a scalar
    `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not (isinstance(value, SkyCoord) and value.isscalar):
            raise ValueError(f'{self.name!r} must be a scalar SkyCoord')


class OneDSkyCoord(RegionAttribute):
    """
    Descriptor class to check that value is a 1D
    `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not (isinstance(value, SkyCoord) and value.ndim == 1):
            raise ValueError(f'{self.name!r} must be a 1D SkyCoord')


class ScalarAngle(RegionAttribute):
    """
    Descriptor class to check that value is a scalar angle, either an
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units.
    """

    def _validate(self, value):
        if isinstance(value, Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')
        else:
            raise ValueError(f'{self.name!r} must be a scalar angle')


class PositiveScalarAngle(RegionAttribute):
    """
    Descriptor class to check that value is a strictly positive
    scalar angle, either an `~astropy.coordinates.Angle` or
    `~astropy.units.Quantity` with angular units.
    """

    def _validate(self, value):
        if isinstance(value, Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')

            if not value > 0:
                raise ValueError(f'{self.name!r} must be strictly positive')
        else:
            raise ValueError(f'{self.name!r} must be a positive scalar angle')


class RegionType(RegionAttribute):
    """
    Descriptor class to check the region type of value.
    """

    def __init__(self, name, regionclass):
        super().__init__(name)
        self.regionclass = regionclass

    def _validate(self, value):
        if not isinstance(value, self.regionclass):
            raise ValueError(f'{self.name!r} must be a '
                             f'{self.regionclass.__name__} object')
