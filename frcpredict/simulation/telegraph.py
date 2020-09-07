import numpy as np

from frcpredict.util.numba_compat import jit, prange

# https://en.wikipedia.org/wiki/Telegraph_process
# The mean and variance listed there don't map precisely on to what
# we're dealing with but are probably useful for understanding
# long time behavior.


@jit(nopython=True, parallel=True)
def make_random_telegraph_data(num_trials=10000, t_on=1.0, t_off=1.0, t_bleach=10.0, t_exp=10.0,
                               p_on=0.5):
    """generate realization of on times from telegraph process with bleaching

    Parameters
    ----------
    num_trials : int, optional
        number of data points to collect
    t_off : float, optional
        characteristic lifetime in off state in units of t_on
    t_bleach : float, optional
        characteristic bleaching timescale in units of t_on
    t_exp : float, optional
        exposure time in units of t_on
    p_on : float in [0, 1], optional
        probability at time 0 of being on

    Returns
    -------
    output_array : 1d ndarray of float
        each entry is fraction of exposure time spent in on state
    """
    output_array = np.zeros(num_trials)
    N_switch_array = np.zeros(num_trials)
    for i in prange(num_trials):
        on_time = 0.0
        t_elapsed = 0.0
        N_switches = -1
        state = np.random.binomial(1, p_on)
        while t_elapsed < t_exp:
            N_switches += 1
            t_eff = t_off + state*(t_on - t_off)
            time_until_switch = np.random.exponential(t_eff)
            on_time += state*(min(time_until_switch, t_exp - t_elapsed))
            state = (state + 1) % 2
            t_elapsed += time_until_switch
        N_switch_array[i] = N_switches  # For some reason this line apparently has to run before
        output_array[i] = on_time       # this line in some configurations?
    lifetimes = np.random.exponential(t_bleach, size=num_trials)
    return np.minimum(output_array, lifetimes), N_switch_array
