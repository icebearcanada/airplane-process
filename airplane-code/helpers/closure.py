# Functions for calculating closure quantities from xspectra (aka visibilities)
#
# Author: Brian Pitzel
# Date created:  19-07-23
# Date modified: 19-07-23


import h5py
import numpy as np
import cupy as cp
import digital_rf
import helpers.utils as utils
import datetime


def closure_angles(now, a3, file, rg=-1, rtype='single'):
    """
    Calculate the closure angles for a minimum set of triple-baseline triangles
    (holding antenna 0 as the reference) for a given UTC second of data.

    Parameters:
    -----------
    now : datetime.datetime object
        Which second of xspectra data to calculate the closure angles for

    a3 : list
        A list of possible baseline combinations in numerical order

    file : h5py.File
        The opened file handler from which to read xspectra data

    rg : float
        If rg == -1, calculate the closure angles for all xspectra in the timestamp
        If rg != -1, calculate the closure angles only at range == rg

    rtype : string
        Defines the return type of this function
        If rtype == 'all', return every closure combination in an array
        If rtype == 'single', return a single closure calculation
        *** Note that if rtype == 'all', the 'rg' input is ignored

    Returns:
    --------
    closure_angles : np.array, shape variable
        -   If rtype == 'all', shape == (N, 36)
            The (N, 36) shape corresponds to ALL N closure angles calcluations for each of the 36 unique
            baseline triangles making up the minimal set of closure angles. 

        -   If rtype == 'single', shape == (1, 36)
            The (1, 36) shape corresponds to a single closure angle for each of the 36 unique baseline
            triangles that make up the minimal set of closure angles.

        The values held in this array are actually the absolute difference between the calculated closure
        angle and zero degrees, zero degrees being the expected value of the closure amplitude.

    """
    combo = []
    b1 = []
    b2 = []
    b3 = []
    for i in range(1, 9):
        for j in range(i+1, 10):
            combo.append(f'0{i}{j}')
            b1.append(f'0{i}')
            b2.append(f'{i}{j}')
            b3.append(f'0{j}')
    closure_angles = list(np.zeros((len(combo))))
    ranges = []
    for i in range(len(combo)):
        try:
            moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
            try:
                #print(a3.index(b1[i]), a3.index(b2[i]), a3.index(b3[i]))
                a = file[f'{moment}/xspectra'][:, a3.index(b1[i])]
                b = file[f'{moment}/xspectra'][:, a3.index(b2[i])]
                c = file[f'{moment}/xspectra'][:, a3.index(b3[i])]
                ranges = file[f'{moment}/rf_distance'][:]
            except KeyError:
                continue
            if rtype == 'all':
                p = np.abs(np.rad2deg(np.angle(a) + np.angle(b) - np.angle(c)))
                p = np.where(p > 180, np.abs(p - 360), p)
                closure_angles[i] = p
            elif rg == -1:
                p = np.abs(np.rad2deg(np.angle(a) + np.angle(b) - np.angle(c)))
                p = np.where(p > 180, np.abs(p - 360), p)
                closure_angles[i] = np.max(p)
            else:
                idxs = np.argwhere(ranges == rg)
                if np.any(idxs):
                    p = np.abs(np.rad2deg(np.angle(a[idxs]) + np.angle(b[idxs]) - np.angle(c[idxs])))
                    p = np.where(p > 180, np.abs(p - 360), p)
                    closure_angles[i] = np.max(p)

        except IOError:
            continue

    return closure_angles, ranges

def closure_amplitudes(now, a3, file, rg=-1, rtype='single'):
    """
    Calculate the closure amplitudes for a minimum set of quad-baseline quadrangles
    (determined according to the diagrammatic process in Blackburn, 2020) for a
    given UTC second of data.

    Parameters:
    -----------
    now : datetime.datetime object
        Which second of xspectra data to calculate the closure amplitudes for

    a3 : list
        A list of possible baseline combinations in numerical order

    file : h5py.File
        The opened file handler from which to read xspectra data

    rg : float
        If rg == -1, calculate the closure amplitudes for all xspectra in the timestamp
        If rg != -1, calculate the closure amplitudes only at range == rg

    rtype : string
        Defines the return type of this function
        If rtype == 'all', return every closure combination in an array
        If rtype == 'single', return a single closure calculation
        *** Note that if rtype == 'all', the 'rg' input is ignored

    Returns:
    --------
    closure_amplitudes : np.array, shape variable
        -   If rtype == 'all', shape == (N, 35)
            The (N, 35) shape corresponds to ALL N closure aamplitude calcluations for each of the 35 unique
            baseline triangles making up the minimal set of closure amplitudes. N is the number of range-doppler
            bins that has a strong signal in the given second

        -   If rtype == 'single', shape == (1, 35)
            The (1, 35) shape corresponds to a single closure angle for each of the 35 unique baseline
            triangles that make up the minimal set of closure angles.

        The values held in this array are actually the absolute difference between the calculated closure
        amplitude and unity, unity being the expected value of the closure amplitude.

    """
    combo = []
    # combos are indexed m-n-p-q
    m_p = []
    n_q = []
    m_q = []
    n_p = []
    count = 0
    # see Blackburn et al., 2020, for the diagrammatic procedure to generate this non-redundant set
    loops = [7, 7, 6, 5, 4, 3, 2, 1]
    for i in range(len(loops)):
        for j in range(i+2, i+2+loops[i]):
            if (j+1)%10 == 0:
                # to make the antenna order numbering consistent with 0's, need to put 0 at the start every time.
                # flipping the first and last indices has no effect (e.g. 1290 -> 0291)
                combo.append(f'{(j+1)%10}{i+1}{j}{i}')
                m_p.append(f'{i}{j}')
                n_q.append(f'{(j+1)%10}{i+1}')
                m_q.append(f'{(j+1)%10}{i}')
                n_p.append(f'{i+1}{j}')

            else:
                combo.append(f'{i}{i+1}{j}{(j+1)%10}')
                m_p.append(f'{i}{j}')
                n_q.append(f'{i+1}{(j+1)%10}')
                m_q.append(f'{i}{(j+1)%10}')
                n_p.append(f'{i+1}{j}')

            count += 1

    closure_amplitudes = list(np.zeros((len(combo))))
    ranges = []
    for i in range(len(combo)):
        try:
            moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
            try:
                #print(a3.index(m_n[i]), a3.index(m_p[i]), a3.index(m_q[i]))
                v_mp = file[f'{moment}/xspectra'][:, a3.index(m_p[i])]
                v_nq = file[f'{moment}/xspectra'][:, a3.index(n_q[i])]
                v_mq = file[f'{moment}/xspectra'][:, a3.index(m_q[i])]
                v_np = file[f'{moment}/xspectra'][:, a3.index(n_p[i])]
                ranges = file[f'{moment}/rf_distance'][:]
            except KeyError:
                continue
            if rtype == 'all':
                m = (np.absolute(v_mq) * np.absolute(v_np)) / (np.absolute(v_mp) * np.absolute(v_nq))
                closure_amplitudes[i] = np.abs(m-1)
            elif rg == -1:
                m = (np.absolute(v_mq) * np.absolute(v_np)) / (np.absolute(v_mp) * np.absolute(v_nq))
                closure_amplitudes[i] = np.max(np.abs(m-1))
            else:
                idxs = np.argwhere(ranges == rg)
                if np.any(idxs):
                    m = (np.absolute(v_mq[idxs]) * np.absolute(v_np[idxs])) / (np.absolute(v_mp[idxs]) * np.absolute(v_nq[idxs]))
                    closure_amplitudes[i] = np.max(np.abs(m-1))

        except IOError:
            continue

    return closure_amplitudes, ranges

def closure_quantities(config, time, t, filepath, rg=-1, rtype='single'):
    """
    Calculates the closure amplitudes and closure angles for a given second of UTC data.

    Parameters:
    -----------
    config : Config object
        The Config object (from utils.py) used for the processing

    time : Time object
        The Time object (from utils.py) used for the processing

    t : int
        The second at which to calculate the closure quantities in seconds past epoch

    filepath : string
        The filepath to the level1 data holding the xspectra needed to calculate the closures

    rg : float, optional
        If rg is provided, the closures will be calculated at only the range == rg.
        If rg is not provided, the closures will be calculated at every range and the worst performing
            closures will be returned

    rtype : string
        Defines the return type of this function
        If rtype == 'all', return every closure combination in an array
        If rtype == 'single', return a single closure calculation
        *** Note that if rtype == 'all', the 'rg' input is ignored

    Returns:
    --------
    c_angles : np.array, shape (1, 36)
        A closure phase value for each independent baseline triangle

    c_amplitudes : np.array, shape (1, 35)
        A closure amplitude value for each independent baseline quadrangle
    """
    now = time.get_date(t)
    try:
        file = h5py.File(f'{filepath}'
                         f'{config.radar_name}_{config.experiment_name}_'
                         f'{int(config.snr_cutoff_db):02d}dB_{config.incoherent_averages:02d}00ms_'
                         f'{int(now.year):04d}_'
                         f'{int(now.month):02d}_'
                         f'{int(now.day):02d}_'
                         f'{int(now.hour):02d}_'
                         f'{config.tx_site_name}_{config.rx_site_name}.h5', 'r')
    except IOError:
        print("file error in closure_amplitudes. returning nothing")
        print(filepath)
        return

    a1 = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,7,7,8]
    a2 = [1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,3,4,5,6,7,8,9,4,5,6,7,8,9,5,6,7,8,9,6,7,8,9,7,8,9,8,9,9]
    a3 = []
    for i in range(len(a1)):
        a3.append(f'{a1[i]}{a2[i]}')

    c_angles, ranges = closure_angles(now, a3, file, rg=rg, rtype=rtype)
    c_amplitudes, _ = closure_amplitudes(now, a3, file, rg=rg, rtype=rtype)

    return c_angles, c_amplitudes, ranges

