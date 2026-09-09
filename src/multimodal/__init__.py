"""
Sensor-Aware Multimodal Processing Layer.
Tailored for Chandrayaan-2 OHRC, TMC-2, and IIRS sensors,
projecting disparate radiometric modalities into a shared terrain-aware feature space.
"""
from .sensor_encoder import (
    CommonTerrainRepresentation,
    BaseSensorEncoder,
    OHRCEncoder,
    TMC2Encoder,
    IIRSEncoder,
    GenericLunarEncoder,
    encode_sensor_image
)

__all__ = [
    "CommonTerrainRepresentation",
    "BaseSensorEncoder",
    "OHRCEncoder",
    "TMC2Encoder",
    "IIRSEncoder",
    "GenericLunarEncoder",
    "encode_sensor_image",
]