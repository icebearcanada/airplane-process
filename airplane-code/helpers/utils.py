# Selected util functions from icebear/utils
# Modified 2023-06-07

import yaml
import numpy as np
import cupy as cp
import os
import re
import datetime
from dateutil.tz import tzutc

class Config:
    def __init__(self, configuration):
        self.update_config(configuration)
        # Add version attribute
        here = os.path.abspath(os.path.dirname(__file__))
        regex = "(?<=__version__..\s)\S+"
#        with open(os.path.join(here, '__init__.py'), 'r', encoding='utf-8') as f:
#            text = f.read()
#        match = re.findall(regex, text)
#        setattr(self, 'version', str(match[0].strip("'")))
        # Add date_created attribute
        now = datetime.datetime.now()
        setattr(self, 'date_created', [now.year, now.month, now.day])

    def update_config(self, file):
        if file.split('.')[1] == 'yml':
            with open(file, 'r') as stream:
                cfg = yaml.full_load(stream)
                for key, value in cfg.items():
                    setattr(self, key, np.array(value))
        if file.split('.')[1] == 'h5':
            stream = h5py.File(file, 'r')
            for key in list(stream.keys()):
                if key == 'data' or key == 'coeffs':
                    pass
                # This horrible little patch fixes strings to UTF-8 from 'S' when loaded from HDF5's
                # and removes unnecessary arrays
                elif '|S' in str(stream[f'{key}'].dtype):
                    temp_value = stream[f'{key}'][()].astype('U')
                    if len(temp_value) == 1:
                        temp_value = temp_value[0]
                    setattr(self, key, temp_value)
                else:
                    temp_value = stream[f'{key}'][()]
                    try:
                        if len(temp_value) == 1:
                            temp_value = temp_value[0]
                        setattr(self, key, temp_value)
                    except:
                        setattr(self, key, temp_value)

    def print_attrs(self):
        print("experiment attributes loaded: ")
        for item in vars(self).items():
            print(f"\t-{item}")
        return None

    def update_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def check_attr(self, key):
        if hasattr(self, key):
            return True
        else:
            return False

    def compare_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            if getattr(self, key) == value:
                return True
            else:
                return False

    def add_attr(self, key, value):
        if self.check_attr(key):
            print(f'ERROR: Attribute {key} already exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def remove_attr(self, key):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            delattr(self, key)
        return None


class Time:
    def __init__(self, start, stop, step):
        """
        Class which hold the iteration time series in both human readable and seconds since epoch (1970-01-01) formats.

        Parameters
        ----------
            start : list int
                Start point of time series in format [year, month, day, hour, minute, second, microsecond]
            stop : list int
                Stop point of time series in format [year, month, day, hour, minute, second, microsecond]
            step : list int
                Step size of time series in format [day, hour, minute, second, microsecond]
        """
        if len(start) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(stop) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(step) != 5:
            raise ValueError('Must include [day, hour, minute, second, microsecond]')
        self.start_human = datetime.datetime(year=int(start[0]), month=int(start[1]), day=int(start[2]), hour=int(start[3]),
                                             minute=int(start[4]), second=int(start[5]), microsecond=int(start[6]), tzinfo=tzutc())
        self.stop_human = datetime.datetime(year=int(stop[0]), month=int(stop[1]), day=int(stop[2]), hour=int(stop[3]),
                                            minute=int(stop[4]), second=int(stop[5]), microsecond=int(stop[6]), tzinfo=tzutc())
        self.step_human = datetime.timedelta(days=int(step[0]), hours=int(step[1]), minutes=int(step[2]),
                                             seconds=int(step[3]), microseconds=int(step[4]))
        self.start_epoch = self.start_human.timestamp()
        self.stop_epoch = self.stop_human.timestamp()
        self.step_epoch = self.step_human.total_seconds()

    def get_date(self, timestamp):
        return datetime.datetime.fromtimestamp(timestamp, tz=tzutc())


def unmatched_filtering(samples, code, code_length, nrng, decimation_rate, navg):
    """
    Done on the GPU. Apply the spread spectrum unmatched filter and decimation to the signal. Essentially this
    first decimates the input signal then applies a 'matched filter' like correlation using a
    special psuedo-random code which has been upsampled to match the signal window and contains
    amplitude filtering bits. This essentially demodulates the signal and removes our code.

    See Huyghebaert, (2019). The Ionospheric Continuous-wave E-region Bistatic Experimental
    Auroral Radar (ICEBEAR). https://harvest.usask.ca/handle/10388/12190

    Parameters
    ----------
    samples : complex64 ndarray
        Antenna complex magnitude and phase voltage samples.
    code : float32 ndarray
        Transmitted pseudo-random code sequence.
    code_length : int
        Length of the transmitted psuedo-random code sequence.
    nrng : int
        Number of range gates being processed. Nominally 2000.
    decimation_rate : float32
        Decimation rate (typically 200) to be used by GPU processing, effects Doppler resolution.
    navg : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.

    Returns
    -------
    filtered : complex64 ndarray
        Output decimated and unmatched filtered samples.
    """
    input_samples = np.lib.stride_tricks.as_strided(samples,
                                                    (navg, int(code_length / decimation_rate), nrng, decimation_rate),
                                                    strides=(code_length * samples.strides[0],
                                                             decimation_rate * samples.strides[0], samples.strides[0],
                                                             samples.strides[0]))
    code_samples = windowed_view(code, window_len=decimation_rate, step=decimation_rate)
    return np.einsum('lijk,ik->lji', input_samples, np.conj(code_samples), optimize='greedy')


def unmatched_filtering_v2(samples, code, code_length, nrng, decimation_rate, navg):
    """
    Returns one variance sample instead of 10. Should use significantly less memory
    """
    input_samples = cp.lib.stride_tricks.as_strided(samples,
                                                    (int(code_length / decimation_rate), nrng, decimation_rate),
                                                    strides=(decimation_rate * samples.strides[0], samples.strides[0],
                                                             samples.strides[0]))
    code_samples = windowed_view(code, window_len=decimation_rate, step=decimation_rate)
    return cp.einsum('ijk,ik->ji', input_samples, cp.conj(code_samples), optimize='greedy')


def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.
    :param      ndarray:     The input ndarray
    :type       ndarray:     ndarray
    :param      window_len:  The window length(filter length)
    :type       window_len:  int
    :param      step:        The step(dm rate)
    :type       step:        int
    :returns:   The array with a new view.
    :rtype:     ndarray
    """

    nrows = ((ndarray.shape[-1] - window_len) // step) + 1
    last_dim_stride = ndarray.strides[-1]
    new_shape = ndarray.shape[:-1] + (nrows, window_len)
    new_strides = list(ndarray.strides + (last_dim_stride,))
    new_strides[-2] *= step

    return cp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


def generate_bcode(filepath):
    """
       Uses the pseudo-random code file to generate the binary code for signal matching

    todo
        Make function able to take any code length and resample at any rate
    """
    # Array for storing the code to be analyzed
    b_code = np.zeros(20000, dtype=np.float32)

    # Read in code to be tested
    test_sig = np.fromfile(open(str(filepath)), dtype=np.complex64)

    # Sample code at 1/4 of the tx rate
    y = 0
    for x in range(80000):
        if (x + 1) % 4 == 0:
            if test_sig[x] > 0.0:
                 b_code[y] = 1.0
                 y += 1
            else:
                 b_code[y] = -1.0
                 y += 1

    return cp.asarray(b_code)
