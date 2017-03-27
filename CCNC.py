# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:26:03 2017

@author: hum094
"""

"""
Functions related to the loading and processing of CCNC data from DMT

version: 1.1
date: 2017-03-23


Search for "xkcd" to find sections of the code that need attention
    Things to do:
        - simplify the execution functions (breaking up into readable chunks)
        - write netcdf writing function
        - check that you are utilising the functions already written!
        

"""
import sys
import pandas as pd
import numpy as np
import os
import glob
import pickle
import importlib.util
import atmoscripts
import RVI_Underway
import datetime
import argparse


def main():
    '''
    Collection of scripts to concatenate, QA/QC and perform flow calibrations 
    on raw data coming from the CCNC-100 instrument made by Droplet Measurement 
    Technologies. 
    
    Usage:
    python CCNC.py raw_path output_path output_filetype output_time_resolution, 
    filterBool, flow_cal_file
    
    where:
        raw_path (str) - path where raw data files exist
        output_path (str) - path where output data files are written
        output_filetype (str) - either 'hdf', 'h5' or 'netcdf'
        output_time_resolution (str) - resolution of output data. Must be in 
            the form '--#U' where # is a numeral and U is replaced with either
            "S" for seconds, "M" for minutes, "H" for hours, or "D" for days
        filterBool (bool) - apply filtering or just concatenate raw data
        flow_cal_file (str) - file containing datetimes and flow data for flow
            calibrations. This file must be in the same folder as the raw data
        output_file_frequency (str) - describes how the output data is broken
            up for memory management. Options are 'monthly','weekly','daily' or
            'all'

    '''
    # If no input given, show the docstring only
    if len(sys.argv[0:]) <= 1:
        print(main.__doc__)
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", help="path where raw data exists")
    parser.add_argument("-o","--output_path", help="path where output data\
                                files are written")
    parser.add_argument("-ext","--output_file_extension", help="Extension of \
                                the output filetype. Options include 'hdf', \
                                'h5' or 'netcdf'", default='hdf')
    parser.add_argument("-res","--output_time_resolution", help="time  \
                                resolution of output data. Default 1 second",
                                default='1S')
    parser.add_argument("-q","--QCdata",help="Boolean specifying whether to \
                                perform QC actions on data. Default false",
                                default=False, type=bool)
    parser.add_argument("--flow_cal_file", help="file containing datetimes and\
                                flow data for flow calibrations. This file \
                                must be in the same folder as the raw data.")
    parser.add_argument("--output_file_frequency", help="describes how output \
                                data is broken up for memory management. \
                                Options are 'monthly','weekly','daily' or \
                                'all'", default='all')
    parser.add_argument("-r","--reload_from_source", help="forces reloading\
                                data from source csv files rather than loading\
                                from concatenated data. Boolean", default=True)
    parser.add_argument("--atmos_press", help="specify atmospheric pressure of \
                        measurement location for supersaturation correction. ",
                        default=1010)
    parser.add_argument("--cal_press", help="specify atmospheric pressure of \
                        calibration location for supersaturation correction. ",
                        default=830)
    parser.add_argument("--log_filt_file", help="file containing datetimes for\
                        removal. This file must be in the same folder as the \
                        raw data.")
    
    args = parser.parse_args()
    
    print(args)
    # Interpret the user input arguments
    ccn_raw_data_path = args.raw_path
    ccn_output_data_path = args.output_path
    ccn_output_filetype = args.output_file_extension
    output_time_resolution = args.output_time_resolution
    QC = args.QCdata
    flow_cal_file = args.flow_cal_file
    output_file_frequency = args.output_file_frequency
    reload_from_source = args.reload_from_source
    atmos_press = args.atmos_press
    cal_press = args.cal_press
    log_filt_file = args.log_filt_file
    
    # Test that the inputs are the correct format 
    if not os.path.exists(ccn_raw_data_path):
        print('Raw data path does not exist. Exiting')    
        return
    
    if (ccn_output_data_path is None) or (not os.path.exists(ccn_output_data_path)):
        ccn_output_data_path = create_temp_output_directory()
        print('Output data path does not exist. Creating new folder in:')
        print(ccn_output_data_path)
    
    assert ccn_output_filetype.lower() in ['netcdf','h5','hdf'], \
        "output filetype invalid! Please use either 'netcdf', 'h5', or 'hdf'"
        
    if output_time_resolution not in ['--1S','--5S','--10S','--15S','--30S',
                                      '--1M','--2M','--5M','--10M','--15M',
                                      '--20M','--30M','--1H','--2H','--3H',
                                      '--6H','--8H','--12H','--1D']:
        output_time_resolution = '--1S'
        print('No valid output time resolution given, \
                  no time resampling applied')
    output_time_resolution = output_time_resolution[2:] # remove the '--'
    
    assert isinstance(QC,bool), 'QC must be a boolean value. Exiting'
    
    if flow_cal_file is not None:
        assert os.path.isfile(flow_cal_file), "Can't find flow cal file!"
    
    

    # Loaded data and process
    LoadAndProcess(ccn_raw_data_path,
                   ccn_output_data_path,
                   ccn_output_filename = 'CCN',
                   ccn_output_filetype = ccn_output_filetype,
                   timeResolution = output_time_resolution,
                   concat_file_frequency = output_file_frequency,
                   QC = QC,
                   flow_cal_file = flow_cal_file,
                   log_filt_file = log_filt_file,
                   reload_from_source = reload_from_source,
                   atmos_press = atmos_press,
                   cal_press = cal_press
                   )
    
    return
    
    
def LoadAndProcess(CCN_raw_path, 
                   CCN_output_path = None,
                   CCN_output_filetype = 'hdf',
                   filename_base = 'CCN', 
                   force_reload_from_source = False,
                   QC = False, 
                   timeResolution='1S',
                   concat_file_frequency = 'all',
                   mask_period_timestamp_df = None,
                   CCN_flow_cal_file = None,
                   CCN_flow_cal_df = None,
                   CCN_flow_setpt = 500,
                   CCN_flow_polyDeg = 2,
                   calibrate_for_pressure = False,
                   press_cal = 1010,
                   press_meas = 1010,
                   split_by_supersaturation = True
                   ):
    '''
    Loads CCNC data from raw csv files, concatenates, then saved to output 
    files of either hdf or netcdf format.
    Data can then be quality controlled using parameters output by the 
    instrument. 
    If a file containing flow calibration values is provided, it will then do a 
    flow calibration.
    If a file is provided containing logged events for filtering, it will 
    remove these periods
    If requested, it will perform exhaust removal (assuming its on the RVI)
    
        Things to do: #xkcd
        Change time resolution to deal with a matrix input
        enable force reload from source
        split_by_supersaturation
DONE        deal with a mask_df
DONE        deal with a flowcal_df
        pressure calibrations
        calculate uncertainties
        figure out file naming
    '''
    if ccn_output_data_path is None:
        ccn_output_data_path = ccn_raw_data_path
    
    # Concatenate csv files
    concatenate_from_csv(ccn_raw_data_path,
                         ccn_output_data_path,
                         ccn_output_filename,
                         output_time_resolution,
                         concat_file_frequency,
                         ccn_output_filetype,
                         reload_from_source
                         )
    
    # Load data
    ccn_data = load_ccn(ccn_output_data_path,ccn_output_filetype)
    
    # QC data for internal parameters and for changes in SS
    if QC:
        ccn_data = DataQC(ccn_data)
        save_as(ccn_data,ccn_output_data_path,'QC',ccn_output_filetype)
    
    # Perform flow calibration if data is provided
    if flow_cal_file is not None: #xkcd Need to test!
        ccn_data = flow_cal(ccn_data,flow_cal_file,ccn_raw_data_path)
        save_as(ccn_data,ccn_output_data_path,'flowcal',ccn_output_filetype)
    
    # Calibrate supersaturation
    ccn_data = ss_cal(ccn_data, atmos_press, cal_press)
    save_as(ccn_data,ccn_output_data_path,'sscal',ccn_output_filetype)
    
    # Correct for inlet losses #xkcd
#    ccn_data = inlet_corrections(ccn_data, IE)
#    save_as(ccn_data,ccn_output_data_path,'IE',ccn_output_filetype)
    
    # Separate into different supersaturations
    ccn_data = ss_split(ccn_data)
    save_as(ccn_data,ccn_output_data_path,'ss_split',ccn_output_filetype)
        
    # Filter for logged events
    ccn_data = atmoscripts.log_filter(ccn_data,ccn_raw_data_path,log_filt_file)
    save_as(ccn_data,ccn_output_data_path,'logfilt',ccn_output_filetype)
        
    # Filter for exhaust
    
    save_as(ccn_data,ccn_output_data_path,'exhaustfilt',ccn_output_filetype)
    
    # Resample to hourly data and calculate uncertainties
    
    save_as(ccn_data,ccn_output_data_path,'hourly',ccn_output_filetype)
    
    
    return

###############################################################################
### File IO
############################################################################### 

def load_ccn(data_path, filetype):
    ''' 
    Loads data from concatenated data file.
    '''
    os.chdir(data_path)
    # Get most recently updated file:
    filelist = glob.glob('*.'+filetype)
    fname = min(filelist, key=os.path.getctime)
    if filetype in ['hdf','h5']:
        data = pd.read_hdf(fname, key='CCN')
    
    elif filetype in ['netcdf','nc']:
        # xkcd
        data = None
    
    elif filetype == 'csv':
        data = pd.read_csv(fname,skipinitialspace = True)  
    
    return data

def Load_to_NetCDF(
                RawDataPath,
                DestDataPath=None,
                output_h5_filename = 'CCNC', 
                resample_timebase = None, 
                concat_file_frequency = 'all'
                ):
    ''' 
    xkcd Need to write this - CHECK STATUS OF ATMOSCRIPTS FUNCTION THAT DOES THIS
    '''
    return

def Load_to_HDF(
                RawDataPath,
                DestDataPath=None,
                output_h5_filename = 'CCNC', 
                resample_timebase = None, 
                concat_file_frequency = 'all'
                ):
    '''
	Load data from CSV files, concatenate and write to h5 file
    '''
    if DestDataPath is None:
        os.chdir(RawDataPath)
    else:
        os.chdir(DestDataPath)
    
    if (output_h5_filename is None) or (output_h5_filename == ''):
        output_h5_filename = 'CCN'
    
    
    if not glob.glob('*.h5'): 
        
        os.chdir(RawDataPath)
        filelist = glob.glob('*.csv')
        filelist.sort()
        
        filelist_df = pd.DataFrame(filelist, columns=['filenames'])
        
        if concat_file_frequency.lower() == 'monthly':
            print('Concatenating to monthly files')
            periods = get_unique_periods(filelist, concat_file_frequency)
            filelist_df['id'] = get_month_label(filelist)
       
        elif concat_file_frequency.lower() == 'weekly':
            print('Concatenating to weekly files')
            periods = get_unique_periods(filelist, concat_file_frequency)
            filelist_df['id'] = get_week_label(filelist)
        
        elif concat_file_frequency.lower() == 'daily':
            print('Concatenating to daily files')
            periods = get_unique_periods(filelist, concat_file_frequency)
            filelist_df['id'] = get_day_label(filelist)
            
        elif concat_file_frequency.lower() == 'all':
            # Continue as normal
            print('Concatenating all files into a single HDF')
            periods = None
            
        
        else:
            print("Cannot determine what frequency you want the output file")
        
        
        # Iterate through files
        if periods is not None: # when output is being broken up
            periods.sort()
            for i in range(0, len(periods)):
                output_h5_filename_ = output_h5_filename + '_' + str(periods[i])
                filelist_ = list(filelist_df[
                        filelist_df['id'] == periods[i]]['filenames'])
        
                save_ccn_to_hdf(filelist_, output_h5_filename_, resample_timebase)
        else:
            save_ccn_to_hdf(filelist, output_h5_filename, resample_timebase)
        
    # If no destination path given, write files in raw data path, otherwise
    # move to destination path
    if DestDataPath is not None:
        move_files(RawDataPath, DestDataPath, '.h5')
        
    return 

def Load_to_CSV(
                RawDataPath,
                DestDataPath=None,
                output_h5_filename = 'CCNC', 
                resample_timebase = None, 
                concat_file_frequency = 'all'
                ):
    ''' 
    Need to write this XKCD
    '''
    return

def save_as(data,
            save_path,
            filename_appendage = '',
            filetype='hdf'
            ):
    '''
    Saves data to file, reading the original filename, and appending informative
    text to the filename.
    For example
    CCN.h5 becomes CCN_QC.h5
    CCN.netcdf becomes CCN_QC_flowcal.netcdf
    '''
    assert filetype in ['hdf','h5','netcdf','nc','csv'], "Don't recognise \
                        filetype to save to. Please use hdf, h5, netcdf or csv"
    assert save_path is not None, 'You must specify the directory where you \
                        want to save!'
    os.chdir(save_path)
    
    
    if filetype in ['hdf','h5']:
        fname = get_ccn_filenamebase('h5', filename_appendage)
                
        # Save data to file
        data.to_hdf(fname, key='CCN')
    
    elif filetype in ['netcdf','nc']:
        # Get the filename of the most recently created file
        fname = get_ccn_filenamebase('nc', filename_appendage)
        
        # Save data to file
        # xkcd
        
    elif filetype == 'csv':
        fname = get_ccn_filenamebase('csv', filename_appendage)
        
        # Save data to file
        data.to_csv(fname)
        
    return

def save_ccn_to_hdf(filelist, output_h5_filename, resample_timebase = None):
    
    #If previous file exists, append, if not start new
    if os.path.isfile(output_h5_filename +'.h5'): 
        
        files_already_loaded = atmoscripts.read_filelist_from_file(
                                                        'files_loaded.txt')
        #with open('files_loaded.txt', 'rb') as f:
        #    files_already_loaded = pickle.load(f)        
        
        # Get only the new files to be loaded:
        filelist = list(set(filelist).difference(set(files_already_loaded)))
        
        data_new, fname_current = read_ccn_csv(filelist)
      
        data = pd.read_hdf(output_h5_filename +'.h5',key='CCN')
        
        data = data.append(data_new)
    
    else:
        data, fname_current = read_ccn_csv(filelist)
    
    # Drop any duplicates which may be there, based only on the Timestamp
    data = data.reset_index().drop_duplicates(subset='timestamp', keep='last')
    data = data.set_index('timestamp')
    
    # Sort data by ascending time
    data = data.sort_index()
    
    
    data.to_hdf(output_h5_filename +'.h5', key='CCN')
#    if fname_current is not None: 
#    if os.path.isfile(fname_current):    try:
#            os.remove(fname_current)
#        except:
#            # do nothing
#    
    
    # Save the filenames that have been loaded to file for next update    
#    with open('files_loaded.txt', 'wb') as f:
    try:
        files_already_loaded
    except NameError:
        filelist = filelist
    else:
        filelist = filelist + files_already_loaded    
#        pickle.dump(filelist, f)
    atmoscripts.write_filelist_to_file(filelist, 'files_loaded.txt')
    
    if resample_timebase is not None:
        timebase_resampler(data, time_int = resample_timebase)   
        
    return

def move_files(origin_pth, dest_pth, extension):
    os.chdir(origin_pth)
        
    # Get list of files from source directory
    files = glob.glob(extension)
    
    # check destination path exists, if not, create it
    if not os.path.exists(dest_pth):
        os.makedirs(dest_pth)
    for file in files:
        os.system('move '+ file + ' ' + dest_pth)
    return

def check_filelist(filetype, reload_from_source):
    '''
    Checks if previous files have been created. If not, then return true and 
    create the new files. If so, and you've been asked to reload_from_source,
    return true. Otherwise, return false and don't reload the files. 
    '''
    filelist = glob.glob('*'+filetype)
    if len(filelist) > 0 and reload_from_source:
        # Delete files and return
        for file in filelist:
            os.remove(file)
        filelist_empty = True
    elif len(filelist) == 0:
        filelist_empty = True
        
    else:
        filelist_empty = False
    
    return filelist_empty
    
def concatenate_from_csv(
                    CCN_raw_path,
                    CCN_output_path,
                    output_h5_filename= 'CCN',
                    resample_timebase = '1S',
                    concat_file_frequency = 'all',
                    CCN_output_filetype='hdf',
                    reload_from_source = True
                    ):
    '''
    Loads all the data from the csv file and saves them in either netCDF or h5
    formatted data files.
    '''
    os.chdir(CCN_raw_path)
    
    if CCN_output_filetype == 'netcdf':
        filelist_empty = check_filelist('.nc', reload_from_source)
        if filelist_empty:
            Load_to_NetCDF()
    else:
        filelist_empty = check_filelist('.h5', reload_from_source)
        if filelist_empty:
            Load_to_HDF(CCN_raw_path,
                        CCN_output_path,
                        output_h5_filename = output_h5_filename,
                        resample_timebase = resample_timebase,
                        concat_file_frequency = concat_file_frequency)
    return

def read_ccn_csv(filelist):
    
    # Specify the column names once.
    colnames = [
                'Time', 'Current SS', 'Temps Stabilized', 'Delta T', 'T1 Set', 
                'T1 Read', 'T2 Set', 'T2 Read', 'T3 Set', 'T3 Read', 
                'Nafion Set', 'T Nafion', 'Inlet Set', 'T Inlet', 'OPC Set', 
                'T OPC', 'T Sample', 'Sample Flow', 'Sheath Flow', 
                'Sample Pressure', 'Laser Current', 'overflow', 'Baseline Mon',
                '1st Stage Mon', 'Bin #', 'Bin 1', 'Bin 2', 'Bin 3', 'Bin 4 ', 
                'Bin 5', 'Bin 6', 'Bin 7', 'Bin 8', 'Bin 9', 'Bin 10', 
                'Bin 11', 'Bin 12', 'Bin 13', 'Bin 14', 'Bin 15', 'Bin 16', 
                'Bin 17', 'Bin 18', 'Bin 19', 'Bin 20', 'CCN Number Conc', 
                'Valve Set', 'Alarm Code', 'Alarm Sum'
                ]
    delete_temp_files_manually = False
    temp_files_to_del = [] #initialise
    fname_previous = ''
                
      
    # If there are LOTS of files, break up into groups 
    # before combining into the final set (to manage RAM)
    filelim = 50
    if len(filelist) > filelim:
        needs_final_grouping = True
    else:
        needs_final_grouping = False
    
    j1 = int(np.ceil(len(filelist)/filelim)) # get the number of group files
        
    for j in range(0, j1):
        
        i0 = j*filelim
               
        if j == j1-1: # Last group file has different final limit
            i1 = len(filelist)
        else:
            i1 = (j+1)*filelim     
        
        #Initialise
        fname_current = None
        data = pd.read_csv(filelist[0], 
                            names = colnames, 
                            skiprows = range(0,6), 
                            engine='python',
                            skipinitialspace = True, 
                            usecols=range(49)
                            )
        # Read date from csv file 
        data['date'] = str(pd.read_csv(filelist[0], names = ['label', 'date'], 
                            skiprows = range(2,len(data)+6))['date'][1])                                 
        for i in range(i0, i1):
                # Load csv data        
                data_temp = pd.read_csv(filelist[i], 
                            names = colnames, 
                            skiprows = range(0,6), 
                            engine='python',
                            skipinitialspace = True, 
                            usecols=range(49)
                            )  
                # Read date from csv file 
                data_temp['date'] = str(pd.read_csv(filelist[i], 
                                        names = ['label', 'date'], skiprows = 
                                        range(2,len(data)+6))['date'][1])
    
                #Append new csv file data to existing dataframe
                data = data.append(data_temp)#, ignore_index=True) 
                
                # Save new data to file
                fname_current = 'CCNC_noIndex_temp_'+str(i0)+'to'+str(i+2)+\
                                'of'+str(len(filelist)+1)+'.h5'
                data.to_hdf(fname_current, key='CCN')
                
                # Remove the temporary file    
                if os.path.isfile(fname_previous):
                    try:
                        os.remove(fname_previous)
                    except PermissionError:
                        delete_temp_files_manually = True
                        temp_files_to_del.append(fname_previous)
                fname_previous = fname_current
        
        # Create timstamp from date and time columns
        data['timestamp'] = pd.to_datetime(data['date']+' '+data['Time'], 
                                            format = "%m/%d/%y %H:%M:%S")
        
        
        # Drop any duplicates which may be there, based only on the Timestamp
        data = data.drop_duplicates(subset='timestamp', keep='last')
        # Change the index to the timestamp
        data = data.set_index('timestamp')
        
        
        # Save group file
        if needs_final_grouping:
            data.to_hdf('CCN_group_'+str(j+1)+'of'+str(j1)+'.h5', key='CCN')

        # Remove last temporary file
        try:
            os.remove(fname_previous)
        except:
            continue 
        
        
    
    if needs_final_grouping:
        del data, data_temp
        
        for j in range(0, j1):
            data_temp = pd.read_hdf('CCN_group_'+str(j+1)+'of'+str(j1)+'.h5', 
                                    key='CCN')
            try:
                data
            except NameError:
                data = data_temp
            else:
                data = data.append(data_temp)
            # Remove the temporary file    
            if os.path.isfile('CCN_group_'+str(j+1)+'of'+str(j1)+'.h5'):
                os.remove('CCN_group_'+str(j+1)+'of'+str(j1)+'.h5')
    
    # Clean up any files that couldnt be deleted due to being locked previously
    if delete_temp_files_manually:
        for k in range(0,len(temp_files_to_del)):
            os.remove(temp_files_to_del[k])
        
    return data, fname_current

def create_temp_output_directory():
    '''
    Creates a default output directory when one isn't specified
    '''
    # Check if s drive exists (i.e. on the CSIRO VM), if not, put it in 
    # the default drive on a computer
    if os.path.exists('s:'):
        drive = 's:'
    else:
        drive = sys.executable[0:2]
    
    output_dir = drive+ '\\data_ccn'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def delete_previous_output(output_path, output_filetype, filename_base):
    '''
    Looks for previously loaded files in the output path and removes them, 
    ready for the loading them from the raw files again
    '''
    os.chdir(output_path)
    if output_filetype == 'hdf':
        ext = '*.h5'
    elif output_filetype == 'netcdf':
        ext = '*.nc'
    elif output_filetype == 'csv':
        ext = filename_base+'*.csv'
    
    filelist = glob.glob(ext)
    
    if len(filelist) > 0:
        print('Deleting previous output files to reload from source:')
        for file in filelist:
            os.remove(file)
            print(file + ' deleted')
                    
    return

def load_basic_csv(filename = None, path = None, file_FULLPATH=None):
    '''
    Loads flow calibration data (i.e. timestamps and flow from flow checks)
    from file. There is an assumption that the calibration file (in csv) has 
    been created using excel from data extracted from the text log file.
    Input either the full path to the filename, or the folder path and the 
    filename
    '''
    if file_FULLPATH is None:
        msg = 'Please specify either the full path to the flow cal file, or \
               the calibration filename with the folder path'
        assert filename is not None, msg
        assert path is not None, msg
               
    
        file_FULLPATH = path+filename
    
    
    assert os.path.isfile(file_FULLPATH), \
            'Flow cal file does not exists! Exiting'
    
    df = pd.read_csv(file_FULLPATH, delimiter = ',')        
    
    return df

def load_manual_mask(filename = None, path = None, file_FULLPATH=None):
    '''
    Loads manual mask data (i.e. start & end timestamps and event description)
    from file. There is an assumption that the mask file (in csv) has 
    been created using excel from data extracted from the text log file.
    Input either the full path to the filename, or the folder path and the 
    filename
    '''
    df = load_basic_csv(filename, path, file_FULLPATH)
    return df
    
def load_flow_cals(filename = None, path = None, file_FULLPATH=None):
    '''
    Loads flow calibration data (i.e. timestamps and flow from flow checks)
    from file. There is an assumption that the calibration file (in csv) has 
    been created using excel from data extracted from the text log file.
    Input either the full path to the filename, or the folder path and the 
    filename
    '''
    df = load_basic_csv(filename, path, file_FULLPATH)
    return df

def get_week_label(filelist):
    # Extract the week number of each dates in the filelist
    
    dates = [f[13:19] for f in filelist]
    weeknum = [str(datetime.date(2000+int(day[0:2]),
               int(day[2:4]),int(day[4:6])).isocalendar()[1]) for day in dates]
    year = [str(datetime.date(2000+int(day[0:2]),int(day[2:4]),
               int(day[4:6])).isocalendar()[0]) for day in dates]
    #week_label = ['_wk' + str(s) for s in list(weeknum)]
    week_label = [x+'_wk'+y for x,y in zip(year,weeknum)]
    return week_label
    
def get_day_label(filelist):
    return [f[13:19] for f in filelist]
    
def get_month_label(filelist):
    return [f[13:17] for f in filelist]
    
def get_year_label(filelist):
    return [f[13:15] for f in filelist]
  
def get_unique_periods(filelist, frequency):
     # Extract all unique values in the filelist (do this using set )    
    months = set(get_month_label(filelist))  
    days = set(get_day_label(filelist))

    # Get the iso week numbers of each of the unique days in the filelist  
    weeklabel = set(get_week_label(filelist))
        
    if frequency == 'monthly':
        return list(months)
    elif frequency == 'daily':
        return list(days)
    elif frequency == 'weekly':
        return list(weeklabel)
    else:
        print('Error in get_unique_periods!')
        return

def get_ccn_filenamebase(ext, appendage):
    '''
    Get's the filename of the most recently created file and produces the new
    filename
    '''
    
    filelist = glob.glob('*.'+ ext)
    
    if len(filelist) == 0:
        return 'CCN_unknown' + appendage + '.' + ext
    else:
        # Get the filename of the most recently created file
        fname_old = max(filelist, key=os.path.getctime).split('.')
        
        # if the version of the file is already there, overwrite
        if appendage in fname_old[0]:
            return fname_old[0] + '.' + fname_old[1]
        else:
            return fname_old[0] + '_' + appendage + '.' + fname_old[1]

###############################################################################
### Calibrations, corrections and quality control
###############################################################################

def DataQC(CCN_data, 
           FlowRatio=10.0,
           T1diffLim=0.25,
           T2diffLim=0.25,
           T3diffLim=0.15,
           NafionTdiffLim=0.3,
           OPCT1diffLim=1
           ):
    """
    Filter data that is out of spec. 
    """    
    
    
    ### Flag data to have a closer look at.

    CCNC_data = CCN_data.copy()
    
    # Initialise
    
#    ReviewData = pd.DataFrame(np.nan,index=CCNC_data.index,columns=['review'])
#    ReviewData['CCN Number Conc'] = CCNC_data['CCN Number Conc'].copy()
#    # Concentration lower than 10 /cm3 (as per factory setting)
#    #Data4CloserLook['check'].loc[CCNC_data['CCN Number Conc'] >= 10] = np.nan    
#    ReviewData.loc[CCNC_data['CCN Number Conc'] < 10] = -999
    
    # Remove data for 3 minutes after each change in supersaturation.               
    CCNC_data = ss_transition_removal(CCNC_data)
    
    ### Filter primary dataset
    with np.errstate(invalid = 'ignore'): # Ignore error warnings caused by 
                                          # arithmetic on nans
    
        # Alarms detected by software - note many of these only activate 
        # after a certain time period in the CCNC software
        CCNC_data.loc[CCNC_data['Alarm Code'] > 0] = np.nan
    
        # Concentration lower than 10 /cm3 (as per factory setting)
        CCNC_data.loc[CCNC_data['CCN Number Conc'] < 10] = np.nan
                     
        # Concentration higher than 5000/cm3, the column experiences water 
        # vapor depletion and thus undercounts, see Latham & Nenes, AS&T, 2011
        CCNC_data.loc[CCNC_data['CCN Number Conc'] > 5000] = np.nan
        
        # Flow ratio outside 10 +/- 0.4
        CCNC_data['Flow Ratio'] = \
                        CCNC_data['Sheath Flow'] / CCNC_data['Sample Flow']
        CCNC_data.loc[CCNC_data['Flow Ratio'] > (FlowRatio + 2)]= np.nan
        CCNC_data.loc[CCNC_data['Flow Ratio'] < (FlowRatio - 2)]= np.nan
        
        # Irrelevant SuperSaturation values
        CCNC_data.loc[CCNC_data['Current SS'] < 0] = np.nan
        
        # 80 < Laser current < 120
        CCNC_data.loc[CCNC_data['Laser Current'] > 120] = np.nan
        CCNC_data.loc[CCNC_data['Laser Current'] <  80] = np.nan
        
        # 1st Stage Mon > 4.7 V 
        CCNC_data.loc[CCNC_data['Laser Current'] > 120] = np.nan    
        
        # Temperatures deviating from their setpoints       
        CCNC_data.loc[abs(CCNC_data['T1 Set'] -                        \
                          CCNC_data['T1 Read']) > T1diffLim] = np.nan
        CCNC_data.loc[abs(CCNC_data['T2 Set'] -                        \
                          CCNC_data['T2 Read']) > T2diffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['T3 Set'] -                        \
                          CCNC_data['T3 Read']) > T3diffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['Nafion Set'] -                    \
                          CCNC_data['T Nafion']) > NafionTdiffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['OPC Set'] -                       \
                          CCNC_data['T OPC']) > 1]= np.nan 
    
    return CCNC_data#, ReviewData)

def filter_ccn_stat(data, std_lim = 150, removeData = False):
    '''
    Filters CCN based on statistical standard deviation filter.
    '''
    if removeData:
        data.loc[data['ccn_std'] > std_lim] = np.nan
    else:
        data.loc[data['ccn_std'] > std_lim,'ccn_std_mask'] = np.nan
    return data

def flow_cal(data, 
             flow_cal_filename = None,
             flow_cal_path = None,
             measured_flows_df = None, 
             set_flow_rate = 500, #ccm
             polydeg=2):
    ''' Calibrates CPC_data for measured flow rates. Data input can either be 
    an already loaded dataframe, or directions to the csv file which contains
    the correctly formatted data.
    Parameters:
     - data - dataframe of raw CPC data
     - flow_cal_filename - str - filename of the flow cal csv data.
     - flow_cal_path - str - path of the flow cal csv data
     - measured_flows_df - a dataframe of the times and measured flow rates 
         used for calibration. See CAPRICORN.py for an example
     - set_flow_rate - the flow rate that the instrument SHOULD be at.
     - polydeg - the degree of the polynomial to fit to the measured data and 
         correct with.
    '''
    
    # Load data from file if required:
    if flow_cal_filename is not None:
        # Assume we're already in the required directory if the path isn't 
        # provided
        measured_flows_df = load_flow_cals(flow_cal_filename, flow_cal_path)
    
    
    # Convert dates to seconds since 1 Jan 2000
    x = (measured_flows_df.index - \
         pd.to_datetime('2000-01-01 00:00:00')).total_seconds()
    y = measured_flows_df['flow rate']
    p = np.poly1d(np.polyfit(x,y,deg=polydeg))
    x_data = (data.index - \
              pd.to_datetime('2000-01-01 00:00:00')).total_seconds()

    data['CCN Number Conc'] = data['CCN Number Conc']/set_flow_rate*p(x_data)
    #plt.plot(x,y,'.',xp,p(xp),'--')
    
    return data

def ss_split(data):
    '''
    Splits the data based on its supersaturation value and removes the 
    transition periods when supersaturations haven't stabilised. 
    '''
    # Get a list of the supersaturations in the file:
    ss_list = data['Current SS'].unique()
        
    d={} #Initialise
    for ss in ss_list:
        if not np.isnan(ss):
            # Create a dictionary to store each separated SS
            d['ccn_' + str(ss)] = data[data['Current SS'] == ss][[\
                             'CCN Number Conc'\
                             ]]
    # Concatenate each df in the dictionary
    split_data = pd.concat(d,axis=1)
    
    # Remove multi-indexing
    split_data.columns = split_data.columns.get_level_values(0)
    
    return split_data

def ss_transition_removal(data):
    '''
    removes data that hasn't stabilised its ss yet due to changing SS setpoint
    '''
    # Create empty boolean column in the dataframe
    data['ss_remove'] = False
    
    i = 1
    while i < len(data):
        # Find index of the change in SS
        ss_prev = data['Current SS'][i-1]
        ss_next = data['Current SS'][i]
        if (ss_prev != ss_next) & (not np.isnan(ss_prev)):
            # Find timestamp of change
            timestamp0 = data.index[i]
            # Find timestamp of end of transition - set to n minutes after 
            # change in set point
            timestamp1= timestamp0+ datetime.timedelta(minutes = 3)
            # Set data within the transition range to nan
            data[(data.index >= timestamp0) & (data.index < timestamp1)] = np.nan
            
            # Start checking again from the end of the removed data
            i = data.index.get_loc(timestamp1)+1
        else:
            i = i+1
        
    
    # delete the temporary flag column
    del data['ss_remove']
    return data

def ss_cal(ccn_data, atmos_press = 1010, cal_press = 830):
    '''
    Calibrates the reported supersaturation level for pressure.
    '''
    if atmos_press == None:
        atmos_press = 1010
        print("Used 1010 mbar as atmospheric pressure at measurement location \
              for supersaturation calibration")
    if cal_press == None:
        cal_press = 830
        print("Assumed most recent calibration for the instrument was done in \
              Boulder and so I've used 830 mbar as calibration pressure for supersaturation calibration")
        
    ccn_data['Current SS'] = ccn_data['Current SS'] + \
                             0.028 * (atmos_press - cal_press)/100
    
    return ccn_data

def filter_uwy(uwy_merge_data,
               uwy_path):
    
    assert isinstance(uwy_path, str), "Please define data path as string"
    if 'mask' not in uwy_merge_data.columns:
        
        RVI_Underway.create_uwy_masks(uwy_path,
                                      apply_mask_to_create_filt_dataset=False)
        
    uwy_merge_data.loc[pd.isnull(uwy_merge_data['mask'])] = np.nan

    return uwy_merge_data

def timebase_resampler(
                      data=0,
                      RawDataPath='',
                      input_h5_filename='',
                      variable='CCN',
                      time_int='default'
                      ):
    ''' 
    Time resampling
    '''
    if time_int == '1S':
        print('No timebase resampling performed. Output base 1 second')
        return # no resampling required!

    
    
    #if no data provided, try to load from file
    if not isinstance(data, pd.DataFrame):
        if (not RawDataPath == '') & (not input_h5_filename == ''):
            os.chdir(RawDataPath)
            if os.path.isfile(input_h5_filename+'.h5'): 
                data = pd.read_hdf(input_h5_filename+'.h5', key=variable)
        else:
            print("Please input either a dataframe or \
                  a datapath and filename where data can be found")
            return
    
    
    # define time resampling intervals unless specified in function input
    if time_int == 'default':
        time_int = ['5S','1Min', '5Min', '10Min', \
                    '30Min', '1H', '3H', '6H', '12H', '1D']    
    
    
    # define MAD calculation
    mad = lambda x: np.fabs(x - x.median()).median() 
    
    # define time resampling intervals
    i_lim = len(time_int)
    sub = data.iloc[:,24:44].copy()
    sub_ccn = data['CCN Number Conc'].copy()
    for i in range(0, i_lim):
        t_int = time_int[i]    
        # Initialise    
        data_resamp = sub.resample(t_int,fill_method=None).median()
        data_resamp['ccn_count'] = sub_ccn.resample(t_int,
                                                    fill_method=None).count()
        data_resamp['ccn_med'] = sub_ccn.resample(t_int,
                                                  fill_method=None).median()
        data_resamp['ccn_mad'] = sub_ccn.resample(t_int,
                                                  fill_method=None).apply(mad)
        data_resamp['ccn_avg'] = sub_ccn.resample(t_int,
                                                  fill_method=None).mean()
        data_resamp['ccn_std'] = sub_ccn.resample(t_int,
                                                  fill_method=None).std()
        
        # Rename the cloud droplet bins so they make sense when the 
        # full data is merged
        data_resamp.rename(columns={'Bin 1': 'CDN Bin 1',
                                    'Bin 2': 'CDN Bin 2',
                                    'Bin 3': 'CDN Bin 3',
                                    'Bin 4': 'CDN Bin 4',
                                    'Bin 5': 'CDN Bin 5',
                                    'Bin 6': 'CDN Bin 6',
                                    'Bin 7': 'CDN Bin 7',
                                    'Bin 8': 'CDN Bin 8',
                                    'Bin 9': 'CDN Bin 9',
                                    'Bin 10': 'CDN Bin 10',
                                    'Bin 11': 'CDN Bin 11',
                                    'Bin 12': 'CDN Bin 12',
                                    'Bin 13': 'CDN Bin 13',
                                    'Bin 14': 'CDN Bin 14',
                                    'Bin 15': 'CDN Bin 15',
                                    'Bin 16': 'CDN Bin 16',
                                    'Bin 17': 'CDN Bin 17',
                                    'Bin 18': 'CDN Bin 18',
                                    'Bin 19': 'CDN Bin 19',
                                    'Bin 20': 'CDN Bin 20'},
                           inplace=True)
        
        # Save to file
        if isinstance(data,pd.DataFrame):        
            outputfilename = variable+'_'+time_int[i]+'.h5'
        else:
            outputfilename = input_h5_filename+'_'+ time_int[i] +'.h5'
        data_resamp.to_hdf(outputfilename, key=variable)
    
    return data_resamp

# if this script is run at the command line, run the main script   
if __name__ == '__main__': 
	main()
    
