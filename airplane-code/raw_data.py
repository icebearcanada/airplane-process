# Examines raw ICEBEAR data (power, phase)
# Author: Brian Pitzel
# Date created: 9 October 2024
# Date modified: 9 October 2024

import datetime
import sys
import getopt
import digital_rf
import numpy as np
import numpy.ma as ma
import helpers.utils as utils
import helpers.processing as proc
from numba import njit, jit, prange
import gc
import matplotlib.pyplot as plt
try:
    import cupy as cp
except ModuleNotFoundError:
    print("cupy not found, using numpy")
    import numpy as cp

# ------- Default command line arguments -------
config_file = '/home/ibp2/cygnus-process/cygnus-code/dat/cygnus_processing.yml'
data_source = '/ext_data2/icebear_3d_data/'
save_loc    = '/home/ibp2/cygnus-process/cygnus-data/'
bcode_file = '/mnt/NAS/processing_code/cygnus-process/icebear-process/dat/pseudo_random_code_test_8_lpf.txt' 
year     = 2023
month    = 1
day      = 1
hour     = 4
minute   = 0
second   = 0
period   = 8 # hours to compute over
plot_cadence_s = 5

n_antennas = 10
n_baselines = int(n_antennas * (n_antennas - 1) / 2)

def coherent_average(data):
    av_data = cp.mean(data)
    magnitude = cp.sqrt(cp.real(av_data) * cp.real(av_data) + cp.imag(av_data) * cp.imag(av_data))
    phase = cp.arctan2(cp.imag(av_data), cp.real(av_data))
    return magnitude, phase


def usage():
    print("usage : %s" % sys.argv[0])
    print("\t --source <filepath> Filepath to the top-level directory of .h5 files to use")
    print("\t -y <year> Year value at which to start the calculations")
    print("\t -m <month> Month value at which to start the calculations")
    print("\t -d <day> Day value at which to start the calculations")
    print("\t -h <hour> Hour value at which to start the calculations")
    print("\t --minute <minute> Minute value at which to start the calculations")
    print("\t -s <second> Second value at which to start the calculations")
    print("\t -p <period> Number of hours to compute over")
    print("\t -i <integration time> Integration time of a data point in seconds")
    print("\t -u <user> Name of the current user on this computer")


def generate_bcode(filepath):
    """
       Uses the pseudo-random code file to generate the binary code for signal matching
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


if __name__ == '__main__':
# ------- Parse command line arguments -------
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'y:m:d:h:s:p:r:u:', ['help', 'source=', 'minute='])
    except:
        usage()
        sys.exit()

    for opt, val in opts:
        if opt == '--help':
            usage()
            sys.exit()
        elif opt == '--source':
            data_source = val
        elif opt == '--config':
            config_file = val
        elif opt == '-y':
            year = int(val)
        elif opt == '-m':
            month = int(val)
        elif opt == '-d':
            day = int(val)
        elif opt == '-h':
            hour = int(val)
        elif opt == '--minute':
            minute = int(val)
        elif opt == '-s':
            second = int(val)
        elif opt == '-p':
            period = int(val)
        elif opt == '-i':
            integration_s = int(val)
        elif opt == '-u':
            save_loc = '/home/' + val + '/cygnus-process/cygnus-data-bcode/'
            config_file = '/home/' + val + '/cygnus-process/cygnus-code/dat/cygnus_processing.yml'

    # ------- Setup rest of config and data reader -------
    config = utils.Config(config_file)
    bcode = generate_bcode(bcode_file)
    processing_start = [year,month,day,hour,minute,second,0]

    # adjust for overnighters
    if hour + period - 1 >= 24: 
        stop_hour = hour+period-1-24
        processing_stop = [year,month,day+1,stop_hour,59,59,0]
    else:
        stop_hour = hour+period-1
        processing_stop  = [year,month,day,stop_hour,59,59,0]
    processing_step  = [0,0,0,plot_cadence_s,0]
    
    time = utils.Time(processing_start, processing_stop, processing_step)
    data_reader = digital_rf.DigitalRFReader(data_source)
    channels = data_reader.get_channels()
    
    if len(channels) == 0:
        print(f' ERROR: No data channels found in {config.processing_source}')
        exit()
    else:
        print('channels acquired:')
        for i in range(len(channels)):
            print(f'\t-{str(channels[i])}')


    # for each plot cadence
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        start_sample = int(t * config.raw_sample_rate)
        step_sample = plot_cadence_s * config.raw_sample_rate + config.number_ranges

        # read in the appropriate amount of data for all antennas
        try:
            data = np.array(
            [
                data_reader.read_vector_c81d(start_sample, step_sample, c)
                for c in channels
            ])
        except:
            print('could not read data')
            continue

        # deal with dropped samples
        if cp.all(cp.real(data) == -32768):
            print(f'dropped sample found, skipping {plot_cadence_s} seconds')
            continue 
        
        # decode data
        #decoded_data = []
        #for i in range(plot_cadence_s*10):
        #    decoded = cp.array( # shape (10, 2000, 100) (n_rx, n_rng, codelen/dec_rate)
        #         [
        #             utils.unmatched_filtering_v2(v[i*config.code_length:(i+1)*config.code_length + config.number_ranges],
        #                 bcode,
        #                 config.code_length,
        #                 config.number_ranges,
        #                 config.decimation_rate,
        #                 config.incoherent_averages)
        #             for v in data
        #         ], dtype=np.complex64)
        #    decoded_data.append(decoded)

        #decoded_data = cp.asarray(decoded_data)
        #decoded_data = decoded_data.get()
        
        samples_to_average = 1000

        magnitude = np.mean(np.abs(data.reshape(data.shape[0], -1, samples_to_average)), axis=2) 
        plotting_time = np.linspace(0, plot_cadence_s, magnitude.shape[1])
        fig, axs = plt.subplots(len(range(n_antennas)))
        for i, a in enumerate(range(n_antennas)):
            axs[i].set_ylabel(f'A{a}')
            axs[i].plot(plotting_time, np.abs(magnitude[i, :]))
        axs[-1].set_xlabel('Time [seconds past timestamp]')
        fig.suptitle(f'Magnitude of Raw Antenna Data: {datetime.datetime.utcfromtimestamp(t)}')
        

        phase = np.angle(np.mean(data.reshape(data.shape[0], -1, samples_to_average), axis=2)) 
        plotting_time = np.linspace(0, plot_cadence_s, phase.shape[1])
        fig, axs = plt.subplots(len(range(n_antennas)))
        for i, a in enumerate(range(n_antennas)):
            axs[i].set_ylabel(f'A{a}')
            axs[i].plot(plotting_time, np.rad2deg(phase[i, :]))
        axs[-1].set_xlabel('Time [seconds past timestamp]')
        fig.suptitle(f'Phase of Raw Antenna Data: {datetime.datetime.utcfromtimestamp(t)}')
        
        



        plt.show()
