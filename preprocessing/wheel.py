
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd

def wheel_to_angles(values_wheel, unwrap=True, ticks_per_rev=600):
    ref = values_wheel[0]
    ticks = np.array(values_wheel) - ref
    angles = (2 * np.pi * ticks / ticks_per_rev) % (2 * np.pi)
    if unwrap:
        return np.unwrap(angles)
    return angles


def average_duplicate_times(times_wheel, values_wheel):
    times_wheel = np.asarray(times_wheel)
    values_wheel = np.asarray(values_wheel)

    # get unique times and inverse indices
    unique_times, inverse_indices = np.unique(times_wheel, return_inverse=True)

    # accumulate sums and counts using bincount
    summed_values = np.bincount(inverse_indices, weights=values_wheel)
    counts = np.bincount(inverse_indices)

    # compute averages
    averaged_values = summed_values / counts

    return unique_times, averaged_values


def restore_missing_wheel_values(values_wheel, times_wheel, wheel_radians_threshold=0.04, restoration_window_ms=1000, ticks_per_rev=600):
    values_wheel = np.asarray(values_wheel)
    times_wheel = np.asarray(times_wheel)

    # convert to unwrapped angles
    angles_unwrapped = wheel_to_angles(values_wheel, unwrap=True, ticks_per_rev=ticks_per_rev)

    # find large jumps (holes)
    dtheta = np.diff(angles_unwrapped)
    jumps = np.where(np.abs(dtheta) > wheel_radians_threshold)[0]

    # prepare full timeline
    full_time_range = np.arange(times_wheel[0], times_wheel[-1] + 1)
    restored_angles = np.full_like(full_time_range, fill_value=np.nan, dtype=np.float64)

    # initially populate known values
    for i, t in enumerate(times_wheel):
        restored_angles[full_time_range == t] = angles_unwrapped[i]

    # interpolate over each hole using local spline
    for idx in jumps:
        t_start = times_wheel[idx]
        t_end = times_wheel[idx + 1]

        win_start = t_start - restoration_window_ms
        win_end = t_end + restoration_window_ms

        # get windowed data
        in_window = (times_wheel >= win_start) & (times_wheel <= win_end)
        if np.sum(in_window) < 4:
            continue # not enough points to fit a spline

        t_win = times_wheel[in_window]
        a_win = angles_unwrapped[in_window]

        spline = UnivariateSpline(t_win, a_win, s=0)

        gap_range = np.arange(t_start + 1, t_end)
        restored_angles[np.isin(full_time_range, gap_range)] = spline(gap_range)

    return restored_angles, full_time_range


# convert raw ticks to angles
def ticks_to_angles(wheel: np.ndarray, ticks_per_rev: int = 600) -> np.ndarray:
    ticks = wheel[:, 1]
    return wheel_to_angles(ticks, unwrap=True, ticks_per_rev=ticks_per_rev)

# find ITI intervals from states
def extract_iti_intervals(states: pd.DataFrame) -> np.ndarray:
    df = states.copy().sort_values('time').reset_index(drop=True)
    inter_idxs = df.index[df['state'] == 'intertrial'].to_numpy()
    hold_idxs = df.index[df['state'] == 'hold'].to_numpy()

    intervals = []
    for i in inter_idxs:
        future_holds = hold_idxs[hold_idxs > i]
        if future_holds.size == 0:
            continue
        j = future_holds[0]
        t0, t1 = df.at[i, 'time'], df.at[j, 'time']
        intervals.append((t0, t1))
    return np.array(intervals)

# interpolate wheel at 1 kHz grid
def interpolate_wheel(wheel_ms: np.ndarray, wheel_pos: np.ndarray):
    t_start, t_end = wheel_ms.min(), wheel_ms.max()
    time_grid = np.arange(t_start, t_end + 1)
    pos_interp = np.interp(time_grid, wheel_ms, wheel_pos)
    return time_grid, pos_interp

# find slowest ITI windows
def find_slowest_windows(wheel_ms: np.ndarray, 
        wheel_pos: np.ndarray,
        intervals: np.ndarray, 
        ITI_WIN: int = 800):
    time_grid, pos_interp = interpolate_wheel(wheel_ms, wheel_pos)
    speed = np.abs(np.gradient(pos_interp, 1) * 1000.0)
    avg_speed = np.convolve(speed, np.ones(ITI_WIN) / ITI_WIN, mode='valid')
    window_starts = time_grid[:len(avg_speed)]

    results = []
    for (t0, t1) in intervals:
        valid = np.where(
            (window_starts >= t0) &
            (window_starts + ITI_WIN - 1 <= t1)
        )[0]
        if valid.size == 0:
            continue
        best_idx = valid[np.argmin(avg_speed[valid])]
        results.append({
            'iti_start': t0,
            'iti_end': t1,
            'win_start': window_starts[best_idx],
            'win_end': window_starts[best_idx] + ITI_WIN - 1,
            'mean_speed': avg_speed[best_idx]
        })

    return pd.DataFrame(results)

# extract wheel-speed features
def wheel_speed_features(slowest_windows: pd.DataFrame, 
        speed: np.ndarray,
        time_grid: np.ndarray, 
        ITI_WIN: int = 800):
    t0_grid = int(time_grid[0])
    feats = []
    for _, r in slowest_windows.iterrows():
        i0 = int(r.win_start) - t0_grid
        i1 = i0 + ITI_WIN
        s = speed[i0:i1]
        if s.size != ITI_WIN:
            continue
        feats.append([
            float(r.mean_speed),
            float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
            float(np.max(s)),
            float(np.percentile(s, 90)),
            float(np.percentile(s, 10)),
        ])
    return np.asarray(feats, dtype=np.float64)
