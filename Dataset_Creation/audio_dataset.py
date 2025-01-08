import dataclasses
import torch
import torchaudio
import numpy as np
from typing import List,Union
from torch.utils.data import Dataset


LABELS = ['angry','joy','love','sad','scary','surprised']


@dataclasses.dataclass
class audio_segment:
    waveform : np.ndarray
    sr : int
    label : str
    org_file : str


class AudioDataset(Dataset):
    def __init__(self, data : Union[List[tuple],str]):
        super().__init__()
        if isinstance(data, str):
            data = torch.load(data,weights_only=False)
        self.org_data = data
        self.data = [audio_segment(*i) for i in data if len(i) == 4]
        self.label_dist = {}
        for i in self.data:
            if i.label not in self.label_dist:
                self.label_dist[i.label] = 1
            else:
                self.label_dist[i.label] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        details = self.data[idx].__dict__.copy()
        del details['waveform']
        del details['label']
        resampler = torchaudio.transforms.Resample(orig_freq=self.data[idx].sr, new_freq=48000)
        waveform = resampler(torch.tensor(self.data[idx].waveform))
        return waveform, self.data[idx].label, details
