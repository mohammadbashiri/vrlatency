import numpy as np
import pandas as pd
from io import StringIO


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


def get_latency(df):
    def detect_latency(df, thresh):
        idx = np.where(df.SensorBrightness > thresh)[0][0]
        return df.Time.iloc[idx] - df.Time.iloc[0]

    latencies = df.groupby('Trial').apply(detect_latency, thresh=df.SensorBrightness.mean())
    latencies.name = 'DisplayLatency'
    return latencies
