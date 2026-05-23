import numpy as np

from samplers import MultiDateBatchSampler


def test_sampler_groups_full_dates():
    date_ids = np.array([1, 1, 2, 2, 2, 3, 4, 4])

    sampler = MultiDateBatchSampler(
        date_ids=date_ids,
        days_per_batch=2,
        shuffle_dates=False,
        drop_last=False,
    )

    batches = list(iter(sampler))

    assert batches == [
        [0, 1, 2, 3, 4],
        [5, 6, 7],
    ]


def test_sampler_len_with_drop_last():
    date_ids = np.array([10, 10, 11, 12, 13, 13])

    sampler = MultiDateBatchSampler(
        date_ids=date_ids,
        days_per_batch=3,
        shuffle_dates=False,
        drop_last=True,
    )

    assert len(sampler) == 1
    assert list(iter(sampler)) == [[0, 1, 2, 3]]


def test_sampler_len_without_drop_last():
    date_ids = np.array([10, 10, 11, 12, 13, 13])

    sampler = MultiDateBatchSampler(
        date_ids=date_ids,
        days_per_batch=3,
        shuffle_dates=False,
        drop_last=False,
    )

    assert len(sampler) == 2
    assert list(iter(sampler)) == [[0, 1, 2, 3], [4, 5]]
