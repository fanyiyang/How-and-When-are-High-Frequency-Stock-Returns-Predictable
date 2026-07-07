"""Regression tests for the clock/window semantics in load_data_and_build_feature.

Run with ``pytest tests/`` or ``python tests/test_clock_windows.py``.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data_and_build_feature import rolling_window_diff, volume_clock  # noqa: E402


def test_volume_windows_are_share_based_not_trade_based():
    """A window of "300 volume" must shrink to 1 trade after a 900-share block trade."""
    qty = pd.Series([50.0, 50, 900, 50, 50])
    volumecum = qty.cumsum()  # 50, 100, 1000, 1050, 1100
    prices = pd.Series([10.0, 11, 12, 13, 14], index=volume_clock(volumecum))

    counts = rolling_window_diff(prices, "0s", "300s", "count")

    # Row-based rolling would give a constant 3 once the window fills up.
    # Share-based windows must "forget" everything older than 300 shares,
    # so right after the 900-share block only that block is in the window.
    expected = [np.nan, 1.0, np.nan, 1.0, 2.0]
    assert counts.isna().tolist() == [np.isnan(v) for v in expected]
    assert counts.dropna().tolist() == [v for v in expected if not np.isnan(v)]


def test_zero_width_left_window_does_not_poison_features():
    """rolling("0ms", closed="left") is all-NaN; the smallest scale must not inherit that."""
    idx = pd.to_datetime(
        ["093000.000", "093000.020", "093000.040", "093000.060"], format="%H%M%S.%f"
    )
    prices = pd.Series([10.0, 11, 12, 13], index=idx)

    breadth = rolling_window_diff(prices, "0ms", "100ms", "count")

    # Every row except the first has trades inside its trailing 100ms window.
    assert breadth.iloc[1:].notna().all()
    assert breadth.iloc[1:].tolist() == [1.0, 2.0, 3.0]


def test_nested_windows_match_direct_computation():
    """[t-400ms, t-200ms) via the diff-of-rollings must equal a direct window sum."""
    times = pd.to_datetime([0, 100, 250, 320, 450, 500], unit="ms")
    values = pd.Series([1.0, 2, 4, 8, 16, 32], index=times)

    got = rolling_window_diff(values, "200ms", "400ms", "sum")

    for t, _ in values.items():
        in_right = values[(values.index >= t - pd.Timedelta("400ms")) & (values.index < t)]
        in_window = values[(values.index >= t - pd.Timedelta("400ms"))
                           & (values.index < t - pd.Timedelta("200ms"))]
        # Convention inherited from the diff-of-rollings construction: NaN when
        # the outer window is empty, plain sum (0 included) otherwise.
        expected = in_window.sum() if len(in_right) else np.nan
        actual = got.loc[t]
        assert (np.isnan(expected) and np.isnan(actual)) or actual == expected


def test_volume_clock_labels_average_future_shares():
    """Return1000Volumes-style label: mean price over the next N shares, not N trades.

    Mirrors response_calculator: data sorted by descending time, rolling over
    the volume clock picks up the *future* window including the current trade.
    """
    # Ascending time: qty 100 each, prices 10,11,12,13; one 5000-share block at the end.
    qty_asc = [100.0, 100, 100, 100, 5000]
    px_asc = [10.0, 11, 12, 13, 20]
    volcum_asc = np.cumsum(qty_asc)  # 100, 200, 300, 400, 5400

    # response_calculator sees the data time-DESCENDING.
    volcum_desc = volcum_asc[::-1]
    px_desc = px_asc[::-1]
    price_on_volume = pd.Series(px_desc, index=volume_clock(volcum_desc))

    label = price_on_volume.rolling("300s").mean().values / np.array(px_desc) - 1

    # For the first trade (volcum=100, price=10) the forward window is
    # volcum in [100, 400) -> prices 10,11,12 -> mean 11.
    first_trade_label = label[-1]
    assert np.isclose(first_trade_label, 11.0 / 10 - 1)
    # For the trade at volcum=300 (price=12) the window is volcum in
    # [300, 600); the 5000-share block sits at volcum 5400, far outside,
    # so only prices 12,13 -> mean 12.5. Trade-count rolling(2) would have
    # pulled the block's price 20 in here.
    third_trade_label = label[2]
    assert np.isclose(third_trade_label, 12.5 / 12 - 1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("all tests passed")
