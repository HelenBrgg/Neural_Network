
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data, target_feature, indices, windowsize):

        self.data = data
        self.target_feature = target_feature
        self.indices = indices
        self.windowsize = windowsize

    def __len__(self):
        return len(self.data) - self.windowsize + 1

    def __getitem__(self, index):
        start_idx = self.indices[index][0]
        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]
        half = round(self.windowsize/2)
        src = sequence[:self.windowsize-1, 1:]
        #src = src.reshape(-1)
        trg = sequence[self.windowsize-1:, :1]
        return src, trg.squeeze(-1)
