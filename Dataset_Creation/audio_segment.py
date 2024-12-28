import dataclasses
import torch

@dataclasses.dataclass
class audio_segment:
    data : torch.Tensor
    label : str
    org_file : str