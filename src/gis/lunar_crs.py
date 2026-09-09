"""
Planetary Coordinate Reference Systems for the Moon.
Complies with IAU/IAG Working Group on Cartographic Coordinates and Rotational Elements (IAU 2000 / 2015).
Uses a Moon mean sphere radius of R = 1,737,400.0 meters.
Never assumes terrestrial WGS84 / EPSG:4326 datums.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from pyproj import CRS, Transformer

# Authentic Lunar Physical Constants (IAU 2000 / 2015)
LUNAR_MEAN_RADIUS_METERS: float = 1737400.0
LUNAR_EQUATORIAL_RADIUS_METERS: float = 1738140.0
LUNAR_POLAR_RADIUS_METERS: float = 1735970.0

@dataclass
class SelenographicPoint:
    lon_deg: float  # East longitude [-180, +180] or [0, 360]
    lat_deg: float  # Selenographic latitude [-90, +90]

@dataclass
class ProjectedPoint:
    x_meters: float
    y_meters: float

class LunarCRS:
    """
    Lunar Coordinate Reference System Manager.
    Supports Equirectangular (equatorial & mid-latitudes) and Polar Stereographic (South & North Pole).
    """
    def __init__(
        self,
        projection_type: str = "EQUIRECTANGULAR",
        center_lon_deg: float = 0.0,
        center_lat_deg: float = 0.0,
        standard_parallel_deg: Optional[float] = None
    ):
        self.projection_type = projection_type.upper()
        self.center_lon = center_lon_deg
        self.center_lat = center_lat_deg
        self.standard_parallel = standard_parallel_deg if standard_parallel_deg is not None else center_lat_deg
        self.radius = LUNAR_MEAN_RADIUS_METERS
        self._init_crs()

    def _init_crs(self):
        # Construct authentic IAU 2000 Lunar Proj string
        if self.projection_type in ["POLAR_STEREOGRAPHIC_SOUTH", "POLAR_SOUTH"]:
            # Lunar South Pole (Chandrayaan-2/3 focus)
            proj_str = (
                f"+proj=stere +lat_0=-90 +lon_0={self.center_lon} +lat_ts=-80 "
                f"+k=1 +x_0=0 +y_0=0 +R={self.radius} +units=m +no_defs"
            )
        elif self.projection_type in ["POLAR_STEREOGRAPHIC_NORTH", "POLAR_NORTH"]:
            # Lunar North Pole
            proj_str = (
                f"+proj=stere +lat_0=90 +lon_0={self.center_lon} +lat_ts=80 "
                f"+k=1 +x_0=0 +y_0=0 +R={self.radius} +units=m +no_defs"
            )
        elif self.projection_type in ["ORTHOGRAPHIC", "ORTHO"]:
            proj_str = (
                f"+proj=ortho +lat_0={self.center_lat} +lon_0={self.center_lon} "
                f"+x_0=0 +y_0=0 +R={self.radius} +units=m +no_defs"
            )
        else:
            # Default: Lunar Equirectangular
            proj_str = (
                f"+proj=eqc +lat_ts={self.standard_parallel} +lat_0={self.center_lat} "
                f"+lon_0={self.center_lon} +x_0=0 +y_0=0 +R={self.radius} +units=m +no_defs"
            )

        self.proj_crs = CRS.from_proj4(proj_str)
        # Geographic selenographic CRS (lon, lat)
        geo_str = f"+proj=longlat +R={self.radius} +no_defs"
        self.geo_crs = CRS.from_proj4(geo_str)

        # Transformers
        self._fwd_transformer = Transformer.from_crs(self.geo_crs, self.proj_crs, always_xy=True)
        self._inv_transformer = Transformer.from_crs(self.proj_crs, self.geo_crs, always_xy=True)

    def forward(self, lon_deg: float, lat_deg: float) -> Tuple[float, float]:
        """Convert selenographic (lon, lat) in degrees to projected (x, y) in meters."""
        x, y = self._fwd_transformer.transform(lon_deg, lat_deg)
        return float(x), float(y)

    def inverse(self, x_meters: float, y_meters: float) -> Tuple[float, float]:
        """Convert projected (x, y) in meters to selenographic (lon, lat) in degrees."""
        lon, lat = self._inv_transformer.transform(x_meters, y_meters)
        return float(lon), float(lat)

    def to_wkt(self) -> str:
        """Export OGC WKT representation for GeoTIFF embedding."""
        return self.proj_crs.to_wkt()