import numpy as np
from torch.utils.data import Sampler

class MultiDateBatchSampler(Sampler):
    """Create batches from complete cross-sections over consecutive trading days.

    Each batch contains all available samples for several dates. This structure
    supports portfolio objectives that require daily cross-sectional weights and
    a short time series of portfolio returns.
    """
    def __init__(self, date_ids: np.ndarray, days_per_batch: int = 20, shuffle_dates: bool = True, drop_last: bool = True):
        self.date_ids = np.asarray(date_ids)
        self.days_per_batch = int(days_per_batch)
        self.shuffle_dates = bool(shuffle_dates)
        self.drop_last = bool(drop_last)

        uniq = np.unique(self.date_ids)
        self.unique_dates = uniq

        # map date -> indices
        self.date_to_indices = {}
        for d in uniq:
            self.date_to_indices[int(d)] = np.where(self.date_ids == d)[0]

    def __iter__(self):
        dates = self.unique_dates.copy()
        if self.shuffle_dates:
            np.random.shuffle(dates)

        # chunk dates into groups
        n = len(dates)
        step = self.days_per_batch
        end = n if not self.drop_last else (n // step) * step

        for i in range(0, end, step):
            chunk = dates[i:i+step]
            batch_idx = []
            for d in chunk:
                batch_idx.extend(self.date_to_indices[int(d)].tolist())
            yield batch_idx

    def __len__(self):
        if self.drop_last:
            return len(self.unique_dates) // self.days_per_batch
        return int(np.ceil(len(self.unique_dates) / self.days_per_batch))
