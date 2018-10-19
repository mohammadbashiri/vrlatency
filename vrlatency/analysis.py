import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec


def read_params(path):
    """ Returns the parameters of experiment data file"""
    with open(path, "r") as f:
        header, body = f.read().split('\n\n')
    return dict([param.split(': ') for param in header.split('\n')])


def read_csv(path):
    """ Returns the data of experiment data file"""
    with open(path, "r") as f:
        header, body = f.read().split('\n\n')
    return pd.read_csv(StringIO(body))


def perc_range(x, perc):
    return perc * np.ptp(x) + np.min(x)


def get_display_latencies(df, thresh=.75):
    latencies = []
    sensorf, timef, trialf = df[['SensorBrightness', 'Time', 'Trial']].values.T
    threshf = perc_range(sensorf, thresh)
    trial_range = np.arange(df.Trial.min(), df.Trial.max()+1)
    for trial in trial_range:
        is_trial = trialf == trial
        sensor = sensorf[is_trial]
        time = timef[is_trial]
        try:
            off_idx = np.where(sensor < threshf)[0][0]
            detect_idx = np.where(sensor[off_idx:] > threshf)[0][0]
            latency = time[detect_idx + off_idx] - time[0]
            latencies.append(latency)
        except IndexError:
            latencies.append(np.nan)

    latencies = pd.Series(data=latencies, name='DisplayLatency', index=trial_range)
    latencies.index.name = 'Trial'
    return latencies


def get_tracking_latencies(df):
    """ Returns the latency values for each trial of a Tracking Experiment"""
    def detect_latency(df, thresh):
        diff = np.diff(df.RigidBody_Position > thresh)
        try:
            idx = np.where(diff != 0)[0][0]
            return df.Time.iloc[idx] - df.Time.iloc[0]
        except IndexError:
            return np.nan

    latencies = df.groupby('Trial').apply(detect_latency, thresh=df.RigidBody_Position.mean())
    latencies.name = 'TrackingLatency'
    return latencies


def get_total_latencies(df):
    """ Returns the latency values for each trial of a Total Experiment"""

    data = df.copy()

    # Make columns with Sensor/LED values used for each trial
    sensors = {False: 'LeftSensorBrightness', True: 'RightSensorBrightness'}
    data['Sensor'] = data.apply(lambda row: row[sensors[row['LED_State']]], axis=1)

    thresh = data[['LeftSensorBrightness', 'RightSensorBrightness']].values.mean()

    # Apply trial-based time series analysis
    trials = data.groupby('Trial')
    latencies = trials.apply(lambda df: (df.Time.iloc[[np.where(df.Sensor > thresh)[0][0]]] - df.Time.iloc[0]).values[0])
    latencies.name = 'TotalLatency'

    # Return dataframe of latencies (Trials x (Group, Latency)
    return latencies


def add_clusters(dd, winsize=30, sse_thresh=.1):
    """Depending on the SSE checks if there are more than one cluster among trials"""
    lowest_delay = dd.loc[dd.Trial == dd[dd.DisplayLatency == dd.DisplayLatency.min()].Trial.values[0], 'TrialTransitionTime'].min()
    lower_bound = -5 if lowest_delay < -5 else lowest_delay
    query = '({} < TrialTransitionTime) & (TrialTransitionTime < {})'.format(lower_bound, lower_bound+ 10)
    # query = '(-5 < TrialTransitionTime) & (TrialTransitionTime < 5)'
    dd2 = dd.query(query)
    ref_trial = dd2[dd2.DisplayLatency == dd2.DisplayLatency.min()]  # Min latency used as reference
    ref_sensor = ref_trial['SensorBrightness'].values

    for trialnum, trial in dd2.groupby('Trial'):
        test_sensor = trial['SensorBrightness'].values
        residuals = compute_sse(test_sensor, ref_sensor, win=winsize)
        minimum = find_global_minimum(residuals)
        residuals = residuals / residuals.max()
        min_sse = residuals[minimum]
        dd.loc[dd.Trial == trialnum, 'Cluster'] = 0 if min_sse < sse_thresh else 1

    return dd


def transform_display_df(df, session, winsize=30, thresh=.75):
    """Return dataframe object needed for the analysis"""
    df['Time'] /= 1000
    df['TrialTime'] = df.groupby('Trial').Time.apply(lambda x: x - x.min())
    df['Sample'] = df.groupby('Trial').cumcount()
    df['Session'] = session
    df['Session'] = pd.Categorical(df['Session'])
    df = df.reindex(['Session', 'Trial', 'Sample', 'Time', 'TrialTime', 'SensorBrightness'], axis=1)
    latencies = get_display_latencies(df, thresh=thresh).to_frame().reset_index()
    dfl = pd.merge(df, latencies, on='Trial')
    dfl['TrialTransitionTime'] = dfl['TrialTime'] - dfl['DisplayLatency']
    dfl['ThreshPerc'] = thresh

    return add_clusters(dfl, winsize=winsize)


def compute_sse(x1, x2, win=30):
    x1_mat = np.ndarray(buffer=x1, shape=(len(x1)-win, win), strides=(8, 8), dtype=x1.dtype)  # Rolling backwards
    error = x1_mat.T - x2[win//2:win//2 + x1_mat.shape[0]]
    sse = np.sum(error ** 2, axis=1) # -win//2
    assert len(sse) > 0
    return sse


def find_global_minimum(x):
    """returns indexed position of the  the global minim of a given signal"""
    dx, ddx = np.diff(x), np.diff(x, 2)
    is_zerocrossing = (dx[1:] * dx[:-1]) < 0
    is_positive_slope = ddx > 0
    is_local_minimum = is_zerocrossing & is_positive_slope
    local_minimum_indices = np.where(is_local_minimum)[0] + 1
    if any(is_local_minimum):
        global_minimum_indices = local_minimum_indices[np.argmin(x[local_minimum_indices])]
    else: # which means that there is no trough
        global_minimum_indices = np.where(x == x.min())[0][0]

    global_minimum_index = int(global_minimum_indices)
    return global_minimum_index


def shift_by_sse(dd, winsize=30):
    """Using Sum of Squared errors between brightness signal of each trial to overlay them"""
    sampling_rate = np.diff(dd.TrialTime.values[:2])[0]

    lowest_delay = dd.loc[dd.Trial == dd[dd.DisplayLatency == dd.DisplayLatency.min()].Trial.values[0], 'TrialTransitionTime'].min()
    lower_bound = -5 if lowest_delay < -5 else lowest_delay
    query = '({} < TrialTransitionTime) & (TrialTransitionTime < {})'.format(lower_bound, lower_bound+ 10)
    # query = '(-5 < TrialTransitionTime) & (TrialTransitionTime < 5)'
    dd2 = dd.query(query)

    ref_trial = dd2[dd2.DisplayLatency == dd2.DisplayLatency.min()]  # Min latency used as reference
    ref_sensor = ref_trial['SensorBrightness'].values

    for trialnum, trial in dd2.groupby('Trial'):
        test_sensor = trial['SensorBrightness'].values
        residuals = compute_sse(test_sensor, ref_sensor, win=winsize)
        minimum = find_global_minimum(residuals)
        offset = minimum - winsize // 2

        dd.loc[dd.Trial == trialnum, 'TrialTransitionTime'] -= offset * sampling_rate

    return dd


def get_average_time_and_brightness(dd, bins=70):
    """Returns an average signal representing the time and the brightness value"""
    time_intervals = pd.cut(dd["TrialTransitionTime"], bins=bins)
    avg_transitiontime = dd.groupby(time_intervals).TrialTransitionTime.apply(np.mean)
    avg_brightness = dd.groupby(time_intervals).SensorBrightness.apply(np.mean)

    return avg_transitiontime.values, avg_brightness.values


def plot_shifted_brightness_over_session(time, sensor_brightness, shift_by, trial_idx, ax=None, **kwargs):
    """creates a plot of all the shifted (overlaid on each other) brightness values over a single session
    Args:
        time: time points as the x-axis
        sesnor_brightness: brightness values
        shift_by (float): a single value which add an offset to the time points
        trial_idx: trial number of every datapoint. This gives the function the flexibility to accept arrays from a
        session with non-equal samples per trial
    """
    ax = ax if ax else plt.gca()
    for trial in trial_idx.unique():
        # ax.scatter(time[trial_idx == trial] + shift_by, sensor_brightness[trial_idx == trial])#, c='r', linewidth=1)#, alpha=.01)
        ax.plot(time[trial_idx == trial] + shift_by, sensor_brightness[trial_idx == trial], linewidth=1, alpha=.01, **kwargs)

    return ax


def plot_brightness_threshold(sensor_brightness, thresh=.75, ax=None):
    """Create a line plot for the threshold values chosen for latecny measurement"""
    ax = ax if ax else plt.gca()
    ax.hlines([perc_range(sensor_brightness, thresh)], *ax.get_xlim(), color='k', label='Threshold', linewidth=2,
              linestyle='dotted')
    return ax


def plot_display_brightness_over_session(trial_time, sensor_brightness, nsamples_per_trial, ax=None):
    """Creates a histograme of the brightness values"""
    ax = ax if ax else plt.gca()
    my_cmap = cm.gray_r
    my_cmap.set_bad(color='w')
    H, xedges, yedges = np.histogram2d(trial_time, sensor_brightness, bins=(nsamples_per_trial, 200))
    H = H.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H, interpolation='nearest', origin='low', cmap=my_cmap, aspect='auto',
               extent=extent, norm=colors.LogNorm())

    return ax


def plot_display_brightness_distribution(sensor_brightness, ax=None):
    """Creates the distribution of the brightness values"""
    ax = ax if ax else plt.gca()
    sns.distplot(sensor_brightness, hist_kws={'color': 'k'}, kde_kws={'alpha': 0}, vertical=True, ax=ax)
    return ax


def plot_display_latency_over_session(trials, latencies, ax=None):
    """Makes a line plot of latencies over the course of a session."""
    ax = ax if ax else plt.gca()
    ax.plot(trials, latencies, c='k', linewidth=.5)
    ax.set(xlabel='Trial number', ylabel='Latency (ms)')
    return ax


def plot_display_latency_distribution(latencies, ax=None):
    """Creates the distribution of the latency values"""
    ax = ax if ax else plt.gca()
    sns.distplot(latencies[np.isnan(latencies) == False],
                 hist=True, color="k", kde_kws={"linewidth": 3, "alpha": 1}, vertical=True)
    return ax


def plot_display_figures(dd):
    """Returns a figure with all info concerning display experiment latencies."""

    session = dd.Session.values[0]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})

    plot_display_brightness_over_session(trial_time=dd['TrialTime'], sensor_brightness=dd['SensorBrightness'],
                                         nsamples_per_trial=dd.groupby('Trial')['DisplayLatency'].agg(len).min(),
                                         ax=ax1)

    mean_latency = dd.groupby('Trial').DisplayLatency.mean().mean()
    plot_shifted_brightness_over_session(time=dd['TrialTransitionTime'], sensor_brightness=dd['SensorBrightness'],
                                         trial_idx = dd['Trial'], shift_by=mean_latency, ax=ax1, c='r')

    plot_brightness_threshold(sensor_brightness=dd['SensorBrightness'], thresh=dd['ThreshPerc'].values[0], ax=ax1)
    plot_display_brightness_distribution(sensor_brightness=dd['SensorBrightness'].values, ax=ax2)
    ax1.set_ylim(*ax2.get_ylim())
    ax2.set(xticklabels='', yticklabels='')
    ax1.set(xlabel='Time (ms)', ylabel='Brightness')

    plot_display_latency_over_session(trials=dd['Trial'], latencies=dd['DisplayLatency'], ax=ax3)
    plot_display_latency_distribution(latencies=dd['DisplayLatency'].values, ax=ax4)
    ax3.set_ylim(*ax4.get_ylim())
    ax4.set(xticklabels='', yticklabels='')
    ax3.set(xlabel='Trial', ylabel='Latency (ms)')

    fig.suptitle(session)
    fig.tight_layout(w_pad=0)
    fig.subplots_adjust(top=.9)

    return fig


def transform_tracking_df(dfd, session):
    dfd['Time'] -= dfd['Time'].min()
    dfd['Time'] *= 1000
    dfd['Sample'] = dfd.groupby('Trial').cumcount()
    dfd['RigidBody_Position_norm'] = dfd['RigidBody_Position'] - dfd.RigidBody_Position.mean()
    dfd['LED_State'] = dfd['RigidBody_Position_norm'].max()
    dfd.loc[dfd['LED_Position'] == 'R', 'LED_State'] = dfd['RigidBody_Position_norm'].min()
    dfd['Session'] = session
    dfd['Session'] = pd.Categorical(dfd['Session'])
    dfd = dfd.reindex(['Session', 'Trial', 'Sample', 'Time', 'RigidBody_Position', 'RigidBody_Position_norm', 'LED_Position', 'LED_State'], axis=1)
    dfl = get_tracking_latencies(dfd).to_frame().reset_index()
    df = pd.merge(dfd, dfl, on='Trial')

    return df



def plot_rb_position(time, position, ax=None):
    ax = ax if ax else plt.gca()
    ax.plot(time, position, c='k')
    ax.scatter(time, position, c='k', alpha=.3)
    return ax


def plot_trial_init(init_times, ax=None):
    ax = ax if ax else plt.gca()
    ymin, ymax = ax.get_ylim()
    ax.vlines(init_times, ymin, ymax, 'r')
    return ax


def plot_led_state(time, state, ax=None):
    ax =ax if ax else plt.gca()
    ax.plot(time, state, c='g')
    return ax


def plot_tracking_latency_over_session(trial, latencies, ax=None):
    ax = ax if ax else plt.gca()
    ax.plot(trial, latencies, color='k')
    return ax


def plot_tracking_latency_distribution(latencies, ax=None):
    """Creates the distribution of the latency values"""
    ax = ax if ax else plt.gca()
    sns.distplot(latencies[np.isnan(latencies) == False],
                 hist=True, color="k", bins=list(range(int(latencies.min()), int(latencies.max()), 1)),
                 kde_kws={"linewidth": 3, "alpha": 1}, vertical=True, ax=ax)
    return ax


def plot_tracking_figures(dd):
    """Returns a figure with all info concerning tracking experiment."""

    session = dd.Session.values[0]


    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 4)
    ax1, ax2, ax3 = plt.subplot(gs[0, :]), plt.subplot(gs[1, :3]), plt.subplot(gs[1, 3])

    plot_rb_position(dd['Time'], dd['RigidBody_Position_norm'], ax=ax1)
    trial_init_time = dd.groupby('Trial').Time.apply(lambda x: x.min()).values
    plot_trial_init(trial_init_time, ax=ax1)
    # plot_led_state(dd['Time'], dd['LED_State'], ax=ax1)
    ax1.set(xlabel='Time (ms)', ylabel='Rigid body position (cm)')

    plot_tracking_latency_over_session(dd['Trial'], dd['TrackingLatency'], ax=ax2)
    ax2.set(xlabel='Trial number', ylabel='Latency (ms)')

    plot_tracking_latency_distribution(dd['TrackingLatency'].values, ax=ax3)
    ax3.set(xticklabels='', yticklabels='')

    fig.suptitle(session)
    fig.tight_layout(w_pad=0)
    fig.subplots_adjust(top=.9)

    return fig

def transform_total_df(dfd, session):
    thresh = .75

    dfd['LED_Position'] = dfd.LED_Position.apply(lambda x: x[2])
    dfd['Time'] /= 1000
    dfd['TrialTime'] = dfd.groupby('Trial').Time.apply(lambda x: x - x.min())
    dfd['Sample'] = dfd.groupby('Trial').cumcount()
    dfd['Session'] = session
    dfd['Session'] = pd.Categorical(dfd['Session'])
    dfd['ThreshPerc'] = thresh
    dfd['SensorBrightness'] = 0
    dfd.loc[dfd['LED_Position'] == 'L', 'SensorBrightness'] = dfd.loc[dfd['LED_Position'] == 'L', 'LeftSensorBrightness']
    dfd.loc[dfd['LED_Position'] == 'R', 'SensorBrightness'] = dfd.loc[dfd['LED_Position'] == 'R', 'RightSensorBrightness']
    del dfd['LeftSensorBrightness']
    del dfd['RightSensorBrightness']
    dfd = dfd.reindex(['Session', 'Trial', 'Sample', 'Time', 'TrialTime', 'SensorBrightness', 'LED_Position', 'ThreshPerc'], axis=1)

    # get latencies for each one of them (to_left and to_right transitions)
    dfl = get_display_latencies(dfd).to_frame().reset_index()

    # merge latencies dataframe with the data dataframe
    df = pd.merge(dfd, dfl, on='Trial')

    df['TrialTransitionTime'] = df['TrialTime'] - df['DisplayLatency']

    return df
