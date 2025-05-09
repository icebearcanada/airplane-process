# Collection of functions used during processing of airplane and ICEBEAR data.
# Date created: 27 Feb 2025
# Date modeified: 27 Feb 2025
# Author: Brian Pitzel

import numpy as np
import pymap3d as pm
import datetime
from datetime import timezone
import h5py
from sklearn.linear_model import LinearRegression
import pickle

# convert feet to meters
FT_TO_M = 1 / 3.281

def find_aer_from_receiver(target_lat, target_lon, target_alt):
    """
    Finds the azimuth, elevation and range of a target or set of targets from the hard coded radar receiver
    Returns angles with respect to North, where East of North (clockwise) is positive.
    """
    # receiver geodetic coordinates
    rx_lat = 52.243 # [deg]
    rx_lon = -106.450 # [deg]
    rx_alt = 0 # [m]
    az, el, slant_range = pm.geodetic2aer(target_lat, target_lon, target_alt,
                                    rx_lat, rx_lon, rx_alt,
                                    ell=pm.Ellipsoid.from_name("wgs84"), deg=True)

    return az, el, slant_range # [deg], [deg], [m]


def calculate_baseline_phase(antenna1, antenna2, target_az, target_el):
    """
    Calculates the phase on the baseline given by antenna1-antenna2 from a point source target at target_az, target_el.

    Parameters:
    -----------
    antenna1 : int
        The first antenna in the baseline

    antenna2 : int
        The second antenna in the baseline

    target_az : float or ndarray
        The azimuth of the point source target in degrees counterclockwise from East

    target_el : float or ndarray
        The elevation of the point source target in degrees up from the horizon

    Returns:
    --------
    phase : float or ndarray
        The baseline (visibility) phase of the point source target

    
    """
    # correct for the receiver heading
    target_az_corrected = target_az + 7
    
    # 3D coordinates of antennas
    wavelength = 6.056
    ant_coords = [  [0.,15.10,73.80,24.2,54.5,54.5,42.40,54.5,44.20,96.9], # x
                [0.,0.,-99.90,0.,-94.50,-205.90,-177.2,0.,-27.30,0.],  # y
                [0.,0.0895,0.3474,0.2181,0.6834,-0.0587,-1.0668,-0.7540,-0.5266,-0.4087]] # z

    # coordinates of antennas 1 and 2
    a1 = np.array([ant_coords[0][antenna1], ant_coords[1][antenna1], ant_coords[2][antenna1]])
    a2 = np.array([ant_coords[0][antenna2], ant_coords[1][antenna2], ant_coords[2][antenna2]])

    # baseline vector
    d = a2 - a1

    # unit vector in the direction of the target
    u = -np.array([
        np.cos(np.deg2rad(target_el)) * np.cos(np.deg2rad(target_az_corrected)),
        np.cos(np.deg2rad(target_el)) * np.sin(np.deg2rad(target_az_corrected)),
        np.sin(np.deg2rad(target_el))
    ])

    # path length difference
    delta_L = np.dot(d, u)

    # phase difference
    phase = 2 * np.pi * delta_L / wavelength

    return phase

def angular_mean_and_stdev(angles):
    """
    Calculates a circular mean and circular standard deviation of angles. Convert each angle into unit vector (cos(angle), sin(angle)),
    then sum the unit vectors and find the arctan of the resulting vector.
    Following https://rosettacode.org/wiki/Averages/Mean_angle and https://en.wikipedia.org/wiki/Directional_statistics#Standard_deviation.
    
    Parameters
    ----------
    angles: float ndarray (N,)
        the angles to average [rad]
    Returns
    -------
    average_angle:
        the average angle [rad]
    stdev:
        the standard deviation of the angles [rad]
    """

    # mean
    x = np.nansum(np.ma.cos(angles))
    y = np.nansum(np.ma.sin(angles))
    average_angle = np.ma.arctan2(y, x)
    # stdev
    n = np.count_nonzero(~np.isnan(angles))
    stdev = np.ma.sqrt(-2 * np.ma.log(np.ma.abs( (1/n) * np.nansum(np.ma.exp(1j * angles)) )))
    return average_angle, stdev


def angular_median(angles):
    """
    Calculates the angular median of a set of angles, assuming they are all within 0-360 degrees.
    If the sorted angles have a gap bigger than 180 degrees, the angle after that gap becomes
    the "first" angle and the angles are reordered starting from the first angle, taking the median
    of that newly ordered sequence.
    Parameters
    ----------
    angles: float ndarray (N,)
        the angles to average [rad]
    Returns
    -------
    average_angle:
        the average angle [rad]
    stdev:
        the standard deviation of the angles [rad]
    """
    # prep the arrays
    gap = np.deg2rad(50)
    angles_copy = np.ma.copy(angles)
    angles_copy = angles_copy[~np.isnan(angles_copy)]
    sorted_angles = np.sort(angles_copy)

    # find the consecutive differences in the angles
    if sorted_angles.shape[0] == 1:
        median = sorted_angles[0]
        return median
    diffs = np.ma.diff(sorted_angles)
    if np.nanmax(diffs) >= gap:
        new_first_idx = np.ma.argmax(diffs) + 1
    else:
        new_first_idx = 0

    
    # make the new set of indices based on the difference being greater than 180 or not
    idxs = []
    idxs.append(np.arange(new_first_idx, np.count_nonzero(~np.isnan(angles)))) #angles.shape[0]))
    idxs.append(np.arange(0, new_first_idx))
    idxs = np.hstack(idxs)

    # take the middle of indices as the median value
    median = sorted_angles[idxs[idxs.shape[0] // 2]]
    
    return median


def retrieve_airplane_data(utc_time, time_filter, bounds):
    """
    Retrieves airplane data from the OpenSky API given a utc time array, a filter into that array
    and latitude/longitude bounds in the form (west, south, east, north)
    """
    
    import traffic
    from traffic.data import opensky
    aircrafts_dbs = []
    
    airplane_start = utc_time[time_filter][0]
    for j in range(utc_time[time_filter].shape[0] - 1):
        this_timestamp = utc_time[time_filter][j]
        next_timestamp = utc_time[time_filter][j+1]
        if next_timestamp - this_timestamp >= datetime.timedelta(seconds=30): # if we are going to move onto another airplane
            airplane_end = this_timestamp
            # get aircraft data for the airplane timeframe
            aircrafts_db = opensky.history(
                                airplane_start,
                                airplane_end,
                                bounds=bounds)
            aircrafts_dbs.append(aircrafts_db)

            airplane_start = next_timestamp

    return aircrafts_dbs
    
def load_airplane_data(filepath):
    """
    Instead of retrieving airplane data from the API, load it from a saved file
    """
    
    file = open(filepath, 'rb')
    aircrafts_dbs = pickle.load(file)
    file.close()
    return aircrafts_dbs

def load_level1_airplane_xspectra(files, t_start, t_end):
    """
    Loads and concatenates a set of level 1 data files into numpy arrays.
    Returns the xspectra and the timestamps corresponding to data between t_start and t_end.
    Note that an internal range bracket is applied to ensure that (nearly) only airplane xspectra are returned

    t_start and t_end must be datetime objects
    """
    print(t_start, t_end)
    
    ti_string = f'{t_start.hour:02d}{t_start.minute:02d}{t_start.second:02d}000'
    tf_string = f'{t_end.hour:02d}{t_end.minute:02d}{t_end.second:02d}000'

    xspectra = []
    time_array = []
    
    for file in files:
        try:
            f = h5py.File(file)
        except Exception as e:
            print(f'Excepted {e}\nContinuing...')
            continue
        
        for key in f['data'].keys():
            # if we are outside the time frame or don't have data_flag, move to the next timestamp
            if key < ti_string or key > tf_string:
                continue

            if not f['data'][key]['data_flag'][:]:
                continue

            # filter the ranges knowing that airplanes generally fall in the rf_distance bracket of [245,260]
            rf_distance = f['data'][key]['rf_distance'][:]
            range_filter = (rf_distance >= 245.0) & (rf_distance <= 260.0)

            # append the xspectra and timestamps using the range filter
            xspectra.append(f['data'][key]['xspectra'][range_filter, :])
            time = f['data'][key]['time'][:]
            hour = time[0]
            minute = time[1]
            second = int(time[2] / 1e3)

            temp_time_array = []
            for target in range(xspectra[-1].shape[0]):
                temp_time_array.append(datetime.datetime(t_start.year, t_start.month, t_start.day, hour, minute, second, tzinfo=timezone.utc))
            time_array.append(np.array(temp_time_array))
        
        f.close()

    # concatenate into a single array
    if xspectra != []:
        xspectra = np.concatenate(xspectra, axis=0)
        time_array = np.concatenate(time_array, axis=0)
        print(xspectra.shape)        
        return xspectra, time_array
    else:
        return 0, 0


def unwrap_phase(phasec, tol):
    # wherever the difference in phase is equal to or greater than 180, ALL values past that point should be adjusted.
    # imagine a ripple effect down the sequence
    phase = np.copy(phasec)
    last_point = False
    while not last_point:
        # unwrap the phase of a time sequence of phases
        phase_diff = np.diff(phase)
        for i in range(phase_diff.shape[0]):
            if phase_diff[i] > 180 - tol:
                phase[i+1:] = phase[i+1:] - 360
                #break
            elif phase_diff[i] < -180 + tol:
                phase[i+1:] = phase[i+1:] + 360
                #break
            if i == phase_diff.shape[0]-1:
                last_point = True        
    return phase

def manually_adjust_phase(vis_phase, year, month, day, airplane_idx):
    # manual adjustments cause unwrapping phase is hard
    adjusted_phase = np.copy(vis_phase)
    
    key = f'{year:04d}-{month:02d}-{day:02d}_{airplane_idx}'
    list_of_adjustments = manual_adjustments[key]
    for adj in list_of_adjustments:
        index = adj[0]
        shift = adj[1]
        adjusted_phase[index:] += shift

    return adjusted_phase

# Function to perform linear regression with a known slope
def linear_regression_with_known_slope2(x, y, m):
    A = np.ones((x.shape[0], 1))
    
    y_adjusted = y - m * x
    
    # Get the y-intercept (b)
    b, residuals, _, _ = np.linalg.lstsq(A, y_adjusted, rcond=None)
    
    return b, residuals


def datetime_to_seconds_since_epoch(datetime_array):
    # Convert each datetime to seconds since epoch
    seconds_since_epoch = np.array([int(dt.timestamp()) for dt in datetime_array])
    return seconds_since_epoch


def save_aircrafts_dbs(aircrafts_dbs, date_str, filtered=False):
    """
    filtered: True if the aircrafts_dbs has been filtered, will save to a different folder
    """
    import pickle
    
    # open a file where you want to store the data
    if not filtered:
        file = open(f'aircrafts_dbs_{date_str}.pckl', 'wb')
    else:
        file = open(f'filtered_aircraft/aircrafts_dbs_filtered_{date_str}.pckl', 'wb')
        
    # dump information to that file
    pickle.dump(aircrafts_dbs, file)
    
    # close the file
    file.close()

    return

def rle(inarray):
    """ 
    run length encoding. Partial credit to R rle function. 
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values) 

    example return:
    rle([True, True, True, False, True, False, False])
    Out[8]: 
    (array([3, 1, 1, 2]),
     array([0, 3, 4, 5]),
     array([ True, False,  True, False], dtype=bool))
    
    """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def filter_aircraft_db(aircrafts_dbs, airplane_xspectra, airplane_echo_time):
    """
    Filters an aircraft database to only contain airplanes that are relevant to ICEBEAR.
    This is performed using various criteria. See function comments for details
    """    

    for db in aircrafts_dbs:
        # db is a database of every airplane in the timeframe.
        if db == None:
            aircrafts_dbs.remove(db)
            continue
            
        db.clean_invalid()
            
        # db[i].data.long/lat is the time series of data for one airplane (i)
        for flight in db: # db[i] is a single airplane in the timeframe.
            # indexes in db continue through multiple airplanes, so need to manually find the indices for this airplane
            start_idx = flight.data.index[0]
            end_idx = flight.data.index[-1]
    
            # we don't want to look at the short airplanes
            if end_idx - start_idx <= 6:
                # this is a horrible line, but just says to remove all entries in db where the icao24 matches the icao24 of this flight
                # i.e., remove this flight
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue

            # sometimes airplane have NaN altitude data
            if np.count_nonzero(~np.isnan(flight.data.altitude)) == 0:
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue
 

            # we can filter based on geographic values. this is a remnant of loading aircraft data with expanded geographic bounds.
            if np.any(flight.data.latitude > 52.25) or np.any(flight.data.latitude < 50.8) \
                or np.any(flight.data.longitude > -106.4) or np.any(flight.data.longitude < -109.375):
                
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue
        
            # from the airplane's geographic latitude/longitude/altitude, find the azimuth/elevation/altitude of the airplane from the receiver
            # this azimuth is measured EAST OF NORTH
            az, el, slant_range = find_aer_from_receiver(flight.data.latitude, flight.data.longitude, flight.data.altitude * FT_TO_M)
            # still east of north, but adjusted to be -180 to 180 instead of 0 to 360
            az = np.where(az > 180.0, az - 360.0, az)

            # based on empirical observations, the airplanes all fall within a fairly narrow az/el volume
            if np.all(el > 25) or np.all(az < -128) or np.all(az > -122):
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue
            
            # now, we can compare this airplane data to baseline data and based on the quality of the comparison, remove some airplanes
            this_airplane_db_time = flight.data.timestamp[:] # .to_pydatetime().replace(tzinfo=timezone.utc)
            this_airplane_echo_indices = np.isin(airplane_echo_time, this_airplane_db_time) # indices in ICEBEAR data where timestamps are in the airplane data time
            this_airplane_xspectra = airplane_xspectra[this_airplane_echo_indices, :] # xspectra corresponding to those indices
            this_airplane_echo_time = airplane_echo_time[this_airplane_echo_indices] # time corresponding to those indices

            # sometimes there is not enough data points in the ICEBEAR data
            if this_airplane_xspectra.shape[0] <= 7:
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue
                
            # sometimes the data is dead with no phase
            if np.median(np.imag(this_airplane_xspectra)) == 0.0:
                db.drop(index=db.data['icao24'][db.data['icao24'] == str(flight.icao24)].index, inplace=True)
                continue

    # done filtering. return the modified aircrafts_db
    return aircrafts_dbs