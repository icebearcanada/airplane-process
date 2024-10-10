# Create a summary of sanitized level2 data in a given time frame
#
# Author: Brian Pitzel
# Date Created: 21 August 2024
# Date Modified: 21 August 2024

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import csv


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
    # get the level 2 files
    y = 2023
    m = 11
    d = 16
    y = 2024
    m = 6
    d = 25
    level2_files = [
            f'/mnt/NAS/cygnus_corrected_L2/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5',
            #f'/mnt/NAS/cygnus_corrected_pi56_L2/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5',
            #f'/mnt/NAS/cygnus_corrected_pi56_L2/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5',
            f'/mnt/NAS/level2_data/{y}/{m:02d}/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5',
            f'/mnt/NAS/uncorrected_L2/{y}_{m:02d}_{d:02d}/ib3d_normal_swht_{y}_{m:02d}_{d:02d}_prelate_bakker.h5',
            ]
    # descriptor for this run
    descriptor = 'Flip antennas 5/6 by 180 deg' #'No corrections on antennas 5 and 6'

    # set up vectorized timestamp converter
    vutcfromtimestamp = np.vectorize(datetime.datetime.utcfromtimestamp)

    # time interval of interest
    #t_start = [2023,12,2,4,50,10]
    #t_end   = [2023,12,2,4,50,30]
    t_start = [2023,11,16,8,30,0]
    t_end   = [2023,11,16,8,40,58]

    t_start = [2024,6,25,8,0,0]
    t_end   = [2024,6,25,8,10,59]
    
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
        print(datetime.datetime.utcfromtimestamp(time[0]), datetime.datetime.utcfromtimestamp(time[-1]))
        utc_time = vutcfromtimestamp(time)
        time_filter = (utc_time < end_time) & (utc_time > start_time) 
        print(utc_time[time_filter][0])
        print(utc_time[time_filter][-1])
        print(longitude[time_filter].shape)	
        velocity_azimuth = f['data']['velocity_azimuth'][:]
        velocity_elevation = f['data']['velocity_elevation'][:]
        velocity_magnitude = f['data']['velocity_magnitude'][:]



        # plot lat/lon
        ax_ll.scatter(longitude[time_filter], latitude[time_filter])#, c=snr_db[time_filter])

        # plot alt/time
        ax_alt.scatter(utc_time[time_filter], altitude[time_filter])#, c=snr_db[time_filter])

        # plot az and el / time
        ax_az.scatter(utc_time[time_filter], azimuth[time_filter])
        ax_el.scatter(utc_time[time_filter], elevation[time_filter])

        # plot 2d unfolded 3d box lat/lon/alt

    ax_ll.legend(['Cygnus A Corrections', 'Manual Phase Calibrations', 'Uncorrected Data'])	
    ax_alt.legend(['Cygnus A Corrections','Manual Phase Calibrations', 'Uncorrected Data'])	
    ax_az.legend(['Cygnus A Corrections', 'Manual Phase Calibrations', 'Uncorrected Data'])	
    ax_el.legend(['Cygnus A Corrections', 'Manual Phase Calibrations', 'Uncorrected Data'])	


    plt.show()
