import dataclasses
import torch
from typing import List,Union
from torch.utils.data import Dataset

@dataclasses.dataclass
class audio_segment:
    waveform : torch.Tensor
    label : str
    org_file : str


class AudioDataset(Dataset):
    def __init__(self, data : Union[List[audio_segment],str]):
        if isinstance(data, str):
            data = torch.load(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        details = self.data[idx].__dict__
        del details['waveform']
        del details['label']
        return self.data[idx].waveform, self.data[idx].label, details
