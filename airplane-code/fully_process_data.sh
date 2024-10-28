#!/bin/bash

# december
year=2024
month=07
day=08

start_hour_L1=2
start_minute_L1=0
end_hour_L1=3
end_minute_L1=0

start_hour_L2=0
start_minute_L2=0
end_hour_L2=23
end_minute_L2=59

level1_processing=false
level2_processing=true
sanitizing=true

raw_data_location='/ext_data/icebear_3d_data/'
level1_data_location='/mnt/NAS/level1_data/'${year}/${month}/
level2_data_location='/mnt/NAS/airplane-data/L2/'

user=$(whoami)

# raw to level 1
if [ "$level1_processing" = true ] ; then
	 /home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/airplane-process/level1_script.py -y ${year} -m ${month} -d ${day} -sh ${start_hour_L1} -sm ${start_minute_L1} -eh ${end_hour_L1} -em ${end_minute_L1} --path-L1 ${level1_data_location} --path-raw ${raw_data_location} --cuda 
fi


# level 1 to level 2
if [ "$level2_processing" = true ] ; then
	/home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/airplane-process/level2_script.py -y ${year} -m ${month} -d ${day} -sh ${start_hour_L2} -sm ${start_minute_L2} -eh ${end_hour_L2} -em ${end_minute_L2} --path-L1 ${level1_data_location} --path-L2 ${level2_data_location} --low-res /mnt/NAS/processing_code/icebear-tools/swht_files/swhtcoeffs_ib3d_2021_10_19_360az_090el_10res_85lmax.h5 --swht /mnt/NAS/processing_code/icebear-tools/swht_files/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5
fi

# sanitize
if [ "$sanitizing" = true ] ; then
	/home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/airplane-process/icebear/imaging/sanitizing.py -d ${year}_${month}_${day} -p ${level2_data_location}	
fi
