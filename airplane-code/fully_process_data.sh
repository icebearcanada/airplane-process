#!/bin/bash

# november day and times
year=2023
month=11
day=16

start_hour_L1=7
start_minute_L1=0
end_hour_L1=15
end_minute_L1=0

start_hour_L2=8
start_minute_L2=30
end_hour_L2=8
end_minute_L2=40

# december day and times
#year=2023
#month=12
#day=02

#start_hour_L1=3
#start_minute_L1=0
#end_hour_L1=14
#end_minute_L1=0

#start_hour_L2=4
#start_minute_L2=50
#end_hour_L2=5
#end_minute_L2=0

# june day and times
year=2024
month=06
day=25

start_hour_L1=0
start_minute_L1=0
end_hour_L1=14
end_minute_L1=0

start_hour_L2=8
start_minute_L2=0
end_hour_L2=8
end_minute_L2=10


level1_processing=false
level2_processing=true
sanitizing=true

raw_data_location='/ext_data2/icebear_3d_data/'
level1_data_location='/mnt/NAS/uncorrected_L1/'
level2_data_location='/mnt/NAS/cygnus_corrected_L2/'

user=$(whoami)

# raw to level 1
if [ "$level1_processing" = true ] ; then
	 /home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/cygnus-process/icebear-process/level1_script.py -y ${year} -m ${month} -d ${day} -sh ${start_hour_L1} -sm ${start_minute_L1} -eh ${end_hour_L1} -em ${end_minute_L1} --path-L1 ${level1_data_location} --path-raw ${raw_data_location} --cuda 
fi


# level 1 to level 2
if [ "$level2_processing" = true ] ; then
	/home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/cygnus-process/icebear-process/level2_script.py -y ${year} -m ${month} -d ${day} -sh ${start_hour_L2} -sm ${start_minute_L2} -eh ${end_hour_L2} -em ${end_minute_L2} --path-L1 ${level1_data_location} --path-L2 ${level2_data_location} --low-res /mnt/NAS/processing_code/icebear-tools/swht_files/swhtcoeffs_ib3d_2021_10_19_360az_090el_10res_85lmax.h5 --swht /mnt/NAS/processing_code/icebear-tools/swht_files/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5
fi

# sanitize
if [ "$sanitizing" = true ] ; then
	/home/${user}/anaconda3/envs/eis/bin/python /mnt/NAS/processing_code/cygnus-process/icebear-process/icebear/imaging/sanitizing.py -d ${year}_${month}_${day} -p ${level2_data_location}	
fi
