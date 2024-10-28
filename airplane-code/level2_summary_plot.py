# Compares level 2 data with airplane location data
#
# Author: Brian Pitzel
# Date Created: 3 October 2024
# Date Modified: 17 October 2024

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import csv
import glob
import traffic
from traffic.data import opensky


SMALL_SIZE = 30 #38
MEDIUM_SIZE = 35 #42
BIGGER_SIZE = 50 #56

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig_width_inches = 17
fig_height_inches = 10

if __name__ == "__main__":
    # get the aircraft data
    bounds = (-109.375, 50.771, -106.0, 52.765) # west, south, east, north

    # get the level 2 files
    y = 2022
    m = 12
    d = 13
    level2_files = [
            f'/mnt/NAS/airplane-data/L2/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5'
            ]
    #level2_files = glob.glob(f'/mnt/NAS/airplane-data/L2/*/ib3d_normal_swht_20*.h5')
    # descriptor for this run
    descriptor = 'Airplane' #'No corrections on antennas 5 and 6'

    # set up vectorized timestamp converter
    vutcfromtimestamp = np.vectorize(datetime.datetime.utcfromtimestamp)

    # time interval of interest
    t_start = [2023,12,13,0,15,0]
    t_end   = [2023,12,13,0,18,0]
    
    t_start = [y,m,d,0,0,0]
    t_end   = [y,m,d,23,59,59]
    
    start_time = datetime.datetime(t_start[0], t_start[1], t_start[2], t_start[3], t_start[4], t_start[5])
    end_time = datetime.datetime(t_end[0], t_end[1], t_end[2], t_end[3], t_end[4], t_end[5])

    # set up axes
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    fig_ll, ax_ll = plt.subplots()
    fig_ll.set_size_inches(fig_width_inches, fig_height_inches)
    ax_ll.set_ylabel('Latitude [deg]')
    ax_ll.set_xlabel('Longitude [deg]')
    ax_ll.set_title(f'Lat and Lon - {descriptor}' )
    stoon_airport = [52.17145, -106.70039]
    icebear_rx = [52.24393, -106.45025]
    icebear_tx = [50.89335, -109.40317]
    ax_ll.scatter(stoon_airport[1], stoon_airport[0], marker='*', c='r', s=40)
    ax_ll.scatter(icebear_rx[1], icebear_rx[0], marker='*', c='r', s=40)
    ax_ll.scatter(icebear_tx[1], icebear_tx[0], marker='*', c='b', s=40)

    fig_alt, ax_alt = plt.subplots()
    fig_alt.set_size_inches(fig_width_inches, fig_height_inches)
    ax_alt.set_ylabel('Altitude [km]')
    ax_alt.set_xlabel('Time')
    ax_alt.set_title(f'Altitude - {descriptor}')
    ax_alt.xaxis.set_major_locator(locator)
    ax_alt.xaxis.set_major_formatter(formatter)

    fig_az, ax_az = plt.subplots()
    fig_az.set_size_inches(fig_width_inches, fig_height_inches)
    ax_az.set_ylabel('Azimuth [deg]')
    ax_az.set_xlabel('Time')
    ax_az.set_title(f'Azimuth - {descriptor}')
    ax_az.xaxis.set_major_locator(locator)
    ax_az.xaxis.set_major_formatter(formatter)

    fig_el, ax_el = plt.subplots()
    fig_el.set_size_inches(fig_width_inches, fig_height_inches)
    ax_el.set_ylabel('Elevation [deg]')
    ax_el.set_xlabel('Time')
    ax_el.set_title(f'Elevation - {descriptor}')
    ax_el.xaxis.set_major_locator(locator)
    ax_el.xaxis.set_major_formatter(formatter)

    for file in level2_files:
        print(file)
        f = h5py.File(file)
        altitude = f['data']['altitude'][:]
        azimuth = f['data']['azimuth'][:]
        doppler_shift = f['data']['doppler_shift'][:]
        elevation = f['data']['elevation'][:]
        latitude = f['data']['latitude'][:]
        longitude = f['data']['longitude'][:]
        rf_distance = f['data']['rf_distance'][:]
        slant_range = f['data']['slant_range'][:]
        snr_db = f['data']['snr_db'][:]
        time = f['data']['time'][:]
        utc_time = vutcfromtimestamp(time)
        time_filter = (utc_time < end_time) & (utc_time > start_time) 
        elevation_filter = elevation < 20
        altitude_filter = altitude < 20
        range_filter = rf_distance < 400
        time_filter = time_filter & range_filter & elevation_filter & altitude_filter
        
        """
        current_minute = utc_time[time_filter][0]
        for j in range(utc_time[time_filter].shape[0] - 1):
            this_timestamp = utc_time[time_filter][j]
            next_timestamp = utc_time[time_filter][j+1]
            if next_timestamp - this_timestamp >= datetime.timedelta(minutes=1):
                
                start_time = current_minute
                current_minute = next_timestamp
                current_minute_incremented = current_minute + datetime.timedelta(minutes=1)
                
                # get aircraft data for the airplane timeframe
                aircrafts_db = opensky.history(
                                    current_minute,
                                    current_minute_incremented,
                                    bounds=bounds)
                
                # plot aircraft tracks
                try:
                    for i in range(len(aircrafts_db)):
                        ax_ll.plot(aircrafts_db[i].data.longitude, aircrafts_db[i].data.latitude)
                    #ax_alt.scatter(aircrafts_db[i].data.time, aircrafts_db[i].data.altitude) # trying to plot the time here
                except Exception:
                    continue
        """
            
        airplane_start = utc_time[time_filter][0]
        for j in range(utc_time[time_filter].shape[0] - 1):
            this_timestamp = utc_time[time_filter][j]
            next_timestamp = utc_time[time_filter][j+1]
            if next_timestamp - this_timestamp >= datetime.timedelta(minutes=1): # if we are going to move onto another airplane
                airplane_end = this_timestamp
                
                # get aircraft data for the airplane timeframe
                aircrafts_db = opensky.history(
                                    airplane_start,
                                    airplane_end,
                                    bounds=bounds)

                # plot aircraft tracks
                try:
                    for i in range(len(aircrafts_db)):
                        ax_ll.plot(aircrafts_db[i].data.longitude, aircrafts_db[i].data.latitude)
                    #ax_alt.scatter(aircrafts_db[i].data.time, aircrafts_db[i].data.altitude) # trying to plot the time here
                except Exception:
                    continue

                airplane_start = next_timestamp



        print(utc_time[time_filter][0])
        print(utc_time[time_filter][-1])
        print(longitude[time_filter].shape)	


        velocity_azimuth = f['data']['velocity_azimuth'][:]
        velocity_elevation = f['data']['velocity_elevation'][:]
        velocity_magnitude = f['data']['velocity_magnitude'][:]

        # plot lat/lon
        ax_ll.scatter(longitude[time_filter], latitude[time_filter], c=time[time_filter])
        
        # plot alt/time
        ax_alt.scatter(utc_time[time_filter], altitude[time_filter])#, c=snr_db[time_filter])

        # plot az and el / time
        ax_az.scatter(utc_time[time_filter], azimuth[time_filter])
        ax_el.scatter(utc_time[time_filter], elevation[time_filter])

        # plot 2d unfolded 3d box lat/lon/alt


    plt.show()
