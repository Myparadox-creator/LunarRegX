"""
Sensor metadata definition and loader for lunar missions.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

@dataclass
class SensorMetadata:
    sensor_name: str = "GENERIC_LUNAR"
    gsd: float = 1.0  # Ground Sampling Distance (m/pixel)
    bit_depth: int = 8  # Native dynamic range (8, 12, 16 bit)
    solar_elevation_deg: Optional[float] = None
    solar_azimuth_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    wavelength_nm: Optional[float] = None
    nodata_value: Optional[float] = None
    bounds: Optional[Dict[str, float]] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorMetadata":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: Path) -> "SensorMetadata":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
