import pandas as pd
import numpy as np
from collections import deque

# parameters
# from Koszeghy et al. 2023
N_TRIALS = 10
CHOICE_COL = "choice"
REWARD_COL = "rewarded"

def add_perceived_probs(
    df: pd.DataFrame,
    init_p: float = np.nan,
    window: int = N_TRIALS,
    choice_col: str = CHOICE_COL,
    reward_col: str = REWARD_COL
) -> pd.DataFrame:
    """
    Adds columns:
        PRPr   – perceived reward-prob for RIGHT  (last `window` R choices mean)
        PRPl   – perceived reward-prob for LEFT   (last `window` L choices mean)
        PRPD   – PRPr − PRPl
        RR_rate – right-response rate in the moving window
    All values are NaN until the corresponding buffer has `window` observations.
    """

    # ensure clean dtypes
    df = df.copy()
    df[choice_col] = ch = df[choice_col].astype(str).values
    df[reward_col] = rew = df[reward_col].astype(bool).values

    # rolling buffers
    r_buf, l_buf = deque(maxlen=window), deque(maxlen=window)

    prpr, prpl, rr_rate = [], [], []
    r_counter = deque(maxlen=window)
    # 1 = right response, 0 = left / NR

    for c, r in zip(ch, rew):
        # update side-specific buffers
        if c == "r":
            r_buf.append(r)
            r_counter.append(1)
        elif c == "l":
            l_buf.append(r)
            r_counter.append(0)
        else:
            r_counter.append(0)

        # perceived reward probs
        prpr.append(np.mean(r_buf) if len(r_buf) == window else init_p)
        prpl.append(np.mean(l_buf) if len(l_buf) == window else init_p)

        # right-response rate over the same window
        rr_rate.append(np.mean(r_counter) if len(r_counter) == window else init_p)

    df["PRPr"]   = prpr
    df["PRPl"]   = prpl
    df["PRPD"]   = df["PRPr"] - df["PRPl"]
    df["RR_rate"] = rr_rate
    return df