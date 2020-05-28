import numpy as np
import matplotlib.pyplot as plt
# if you have numba it will massively speed this simulation up. I was getting
# about 300x improvement. If not, we define a dummy decorator so this will
# run.
try:
    import numba
    jit = numba.jit
except ImportError:
    print("Couldn't import numba. This might be a little slow...")
    def jit(nopython=True):
        def pass_through(f):
            return f
        return pass_through

# https://en.wikipedia.org/wiki/Telegraph_process
# The mean and variance listed there don't map precisely on to what
# we're dealing with but are probably useful for understanding
# long time behavior.


@jit(nopython=True)
def make_random_telegraph_data(num_trials=10000, t_on = 1.0, t_off=1.0, t_bleach=10.0,
                               t_exp=10.0, p_on=.5):
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
    output_array = []
    N_switch_array = []
    for _ in range(num_trials):
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
        output_array.append(on_time)
        N_switch_array.append(N_switches)
    lifetimes = np.random.exponential(t_bleach, size=(num_trials))
    return np.minimum(np.array(output_array), lifetimes), np.array(N_switch_array)


if __name__ == '__main__':
    T_exps = np.concatenate((np.linspace(.025, 2.0, 80),
                             np.linspace(2.0, 100.0, 99)))
    p_on = 1.0
    alpha_tau = 5000
    t_bleach = 20.0
    alpha = alpha_tau / t_bleach
    var_tf_arr = []
    poisson_arr = []
    for T_exp in T_exps:
        data = make_random_telegraph_data(10000, t_off=1.0,
                                          t_bleach=t_bleach,
                                          t_exp=T_exp, p_on=p_on)
        var_tf_arr.append(np.var(data / data.mean()))
        poisson_arr.append(1.0 / (alpha*data.mean()))
    var_tf_arr = np.asarray(var_tf_arr)
    poisson_arr = np.asarray(poisson_arr)
    plt.semilogx(T_exps, var_tf_arr)
    plt.semilogx(T_exps, poisson_arr)
    plt.semilogx(T_exps, var_tf_arr + poisson_arr)
    plt.xlabel('T_exp')
    plt.ylabel(r'$\mathrm{var} (\mathcal{P} / E(\mathcal{P}))$')
    plt.legend(('t_f', 'Poisson', 'total'), loc='best')
    plt.title('Noise vs. exposure time: duty cycle = .5, t_bleach = 20.0')
    plt.show()
