from dataclasses import dataclass


@dataclass
class CameraProperties:
    read_out_noise: float
    quantum_efficiency: float
