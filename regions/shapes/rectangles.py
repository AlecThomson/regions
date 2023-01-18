import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
from IPython import embed

from regions._utils.wcs_helpers import pixel_scale_angle_at_skycoord
from regions.core.attributes import (
    PositiveVectorAngle,
    RegionMetaDescr,
    RegionVisualDescr,
    VectorAngle,
    VectorSkyCoord,
)
from regions.core.core import PixelRegion, SkyRegion
from regions.core.metadata import RegionMeta, RegionVisual


class RectanglePixelRegions(PixelRegion):
    _params = ("centers", "widths", "heights", "angles")

    def __init__(self, centers, widths, heights, angles, meta=None, visual=None):
        self.centers = centers
        self.widths = widths
        self.heights = heights
        self.angles = angles
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return self.widths * self.heights

    def contains(self, pixcoord):
        cos_angle = np.cos(self.angles)
        sin_angle = np.sin(self.angles)
        dx = pixcoord.x - self.centers.x
        dy = pixcoord.y - self.centers.y
        dx_rot = cos_angle * dx + sin_angle * dy
        dy_rot = sin_angle * dx - cos_angle * dy
        in_rect = (np.abs(dx_rot) < self.widths.value * 0.5) & (
            np.abs(dy_rot) < self.heights.value * 0.5
        )
        if self.meta.get("include", True):
            return in_rect
        else:
            return np.logical_not(in_rect)

    def to_sky(self, wcs):
        centers = wcs.pixel_to_world(self.centers.x, self.centers.y)
        _, pixscales, north_angles = pixel_scale_angle_at_skycoord(centers, wcs)
        widths = Angle(self.widths * u.pix * pixscales, "arcsec")
        heights = Angle(self.heights * u.pix * pixscales, "arcsec")
        # Region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angles = self.angles - (north_angles - 90 * u.deg)
        return RectangleSkyRegions(
            centers,
            widths,
            heights,
            angles=angles,
            meta=self.meta.copy(),
            visual=self.visual.copy(),
        )

    @property
    def bounding_box(self):
        raise NotImplementedError

    def to_mask(self, mode="center", subpixels=5):
        raise NotImplementedError

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError


class RectangleSkyRegions(SkyRegion):

    _params = ("centers", "widths", "heights", "angles")
    centers = VectorSkyCoord("The center positions as a |SkyCoord|. ")
    widths = PositiveVectorAngle(
        "The widths of the rectangles (before rotation) " "as a |Quantity| angle."
    )
    heights = PositiveVectorAngle(
        "The heights of the rectangles (before " "rotation) as a |Quantity| angle."
    )
    angles = VectorAngle(
        "The rotation angles measured anti-clockwise as a " "|Quantity| angle."
    )
    meta = RegionMetaDescr("The meta attributes as a |RegionMeta|")
    visual = RegionVisualDescr("The visual attributes as a |RegionVisual|.")

    def __init__(self, centers, widths, heights, angles, meta=None, visual=None):
        self.centers = centers
        self.widths = widths
        self.heights = heights
        self.angles = angles
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return self.widths * self.heights

    def to_pixel(self, wcs):
        centers, pixscales, north_angles = pixel_scale_angle_at_skycoord(
            self.centers, wcs
        )
        widths = (self.widths / pixscales).to(u.pix)
        heights = (self.heights / pixscales).to(u.pix)
        # Region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angles = self.angles + (north_angles - 90 * u.deg)
        return RectanglePixelRegions(
            centers,
            widths,
            heights,
            angles=angles,
            meta=self.meta.copy(),
            visual=self.visual.copy(),
        )

    def contains(self, coord):
        cos_angle = np.cos(self.angles)
        sin_angle = np.sin(self.angles)
        dx = coord.ra - self.centers.ra
        dy = coord.dec - self.centers.dec
        dx_rot = cos_angle * dx + sin_angle * dy
        dy_rot = sin_angle * dx - cos_angle * dy
        in_rect = (np.abs(dx_rot) < self.widths * 0.5) & (
            np.abs(dy_rot) < self.heights * 0.5
        )
        if self.meta.get("include", True):
            return in_rect
        else:
            return np.logical_not(in_rect)


if __name__ == "__main__":
    # A little demo script
    from astropy.io import fits

    rectangles = RectangleSkyRegions(
        centers=SkyCoord([1, 2, 3], [1, 2, 3], unit="deg"),
        widths=np.array([1, 2, 3]) * u.deg,
        heights=np.array([1, 2, 3]) * u.deg,
        angles=np.array([1, 2, 3]) * u.deg,
    )
    print(rectangles)
    coords = SkyCoord([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], unit="deg")
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 10,
            "NAXIS2": 10,
            "CTYPE1": "RA---TAN",
            "CRVAL1": 0,
            "CRPIX1": 5,
            "CDELT1": -0.1,
            "CUNIT1": "deg",
            "CTYPE2": "DEC--TAN",
            "CRVAL2": 0,
            "CRPIX2": 5,
            "CDELT2": 0.1,
            "CUNIT2": "deg",
        }
    )
    wcs = WCS(header)
    print(rectangles.to_pixel(wcs))
    print(rectangles.contains(coords))
