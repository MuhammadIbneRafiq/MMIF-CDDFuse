"""In-memory replacement for utils.dataset.H5Dataset.

The original H5Dataset reopens the HDF5 file on every __getitem__ call, which
serializes training behind file I/O and leaves the GPU idle. The whole patch set
is only ~1.5 GB, so load it once up front and index plain numpy arrays instead.

This also sidesteps DataLoader worker processes entirely (num_workers=0), which
matters on Python 3.14+ where the default start method on Linux changed from
'fork' to 'forkserver' -- forkserver re-imports the main module, and train_snow.py
runs its training at module top level with no __main__ guard.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class InMemoryH5(Dataset):
    def __init__(self, h5file_path):
        with h5py.File(h5file_path, 'r') as f:
            keys = list(f['ir_patchs'].keys())
            self.ir = np.stack([f['ir_patchs'][k][()] for k in keys]).astype(np.float32)
            self.vis = np.stack([f['vis_patchs'][k][()] for k in keys]).astype(np.float32)
        gb = (self.ir.nbytes + self.vis.nbytes) / 1e9
        print(f"loaded {len(self.ir)} patch pairs into RAM ({gb:.2f} GB)")

    def __len__(self):
        return len(self.ir)

    def __getitem__(self, i):
        # same (VIS, IR) ordering as the original H5Dataset
        return torch.from_numpy(self.vis[i]), torch.from_numpy(self.ir[i])
