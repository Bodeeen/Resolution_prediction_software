import numpy as np

from frcpredict.util.numba_compat import jit, prange
from typing import Tuple
# https://en.wikipedia.org/wiki/Telegraph_process
# The mean and variance listed there don't map precisely on to what
# we're dealing with but are probably useful for understanding
# long time behavior.


@jit(nopython=True, parallel=True)
def make_random_telegraph_data(num_trials: int, t_on: float, t_off: float,
                               t_bleach: float, t_exp: float, P_on: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realization of on times from telegraph process with bleaching.

    Parameters
    ----------
    num_trials : int
        number of data points to collect
    t_on : float
        characteristic lifetime in on state
    t_off : float
        characteristic lifetime in off state in units of t_on
    t_bleach : float
        characteristic bleaching timescale in units of t_on
    t_exp : float
        exposure time in units of t_on
    P_on : float in [0, 1]
        probability at time 0 of being on

    Returns
    -------
    output_array : 1d ndarray of float
        each entry is fraction of exposure time spent in on state
    """

    output_array = np.zeros(num_trials)
    N_array = np.zeros(num_trials)
    if t_exp <= 0:
        return output_array, N_array
    
    for i in prange(num_trials):
        on_time = 0.0
        t_elapsed = 0.0
        N_switches = -1
        state = np.random.binomial(1, P_on)
        while t_elapsed < t_exp:
            N_switches += 1
            t_eff = t_off + state*(t_on - t_off)
            time_until_switch = np.random.exponential(t_eff)
            on_time += state*(min(time_until_switch, t_exp - t_elapsed))
            state = (state + 1) % 2
            t_elapsed += time_until_switch
        output_array[i] = on_time
        N_array[i] = N_switches
        
    lifetimes = np.random.exponential(t_bleach, size=num_trials)
    
    return np.minimum(output_array, lifetimes), N_array
