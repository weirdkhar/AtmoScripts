# -*- coding: utf-8 -*-
"""
A collection of useful functions for atmospheric sciences
A complement to atmosplots

Written by Ruhi Humphries, 2016
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
import os
import time
import math

# A function to write a dataset to a NetCDF file
# Developed utilising info from http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html
# and http://salishsea-meopar-tools.readthedocs.io/en/latest/netcdf4/

def log_filter(data, 
               raw_data_path = None, log_filename= None, 
               log_mask_df = None):
    '''
    Function to remove data based on the logged events. The log must be
    formatted as a csv file with the first and second columns as the 
    beginning and end of the data removal period. Additional data (e.g.
    a 3rd column containing a description of why the period is being
    removed) is ignored. Timestamps must be formatted as:
         yyyy-mm-dd HH:SS:MM
    '''
    if log_mask_df is None:
        if os.path.exists(log_filename):
            # if a full path is provided
            log_mask = pd.read_csv(log_filename)
        else: 
            assert raw_data_path is not None, 'No path provided for mask filter' 
            assert log_filename is not None, 'No filename provided for mask filter'
            assert os.path.exists(log_filename), 'Specified mask filter file does \
                                                    not exist'
            # the filename and folder are provided separately.    
            os.chdir(raw_data_path)
            # Load file 
            log_mask = pd.read_csv(log_filename)
                
        
        # check whether there is a header or not
        try: # check if the loaded header is actually a date
            pd.to_datetime(log_mask.columns[0])
            # if it is, reload the data using the header option
            log_mask = pd.read_csv(log_filename, 
                                   header = None, 
                                   names = ['start','end', '']
                                   )
        except ValueError:
            # rename first two columns
            log_mask.columns = ['start','end','']
    else:
        log_mask = log_mask_df
            
    # Parse timestamps
    log_mask.iloc[:,0] = pd.to_datetime(log_mask.iloc[:,0])
    log_mask.iloc[:,1] = pd.to_datetime(log_mask.iloc[:,1])
    # work through mask periods and set values to nan
    for i in range(0,len(log_mask)):
        data.loc[(data.index >= log_mask.iloc[i,0]) & \
                 (data.index < log_mask.iloc[i,1])] \
                 = np.nan
        
    return data

def find_variable_parameter(variable, parameter = 'units'):
    '''
    Finds the parameter of the variable being saved to netCDF format.
    It will first explore a previously generated database, and if not present
    in the database, it will ask the user for input and add that to the database.
    '''
    assert parameter in ['units','long_name'],"parameter not available, please check input"
    
    # Change directory to where this script is, assuming the dict file is here.
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if os.path.isfile('variable_description.csv'):    
        ind = pd.read_csv('variable_description.csv',header=0,index_col=0)
        p = ind[parameter]
        p_dict = p.to_dict()
    else:
        p_dict = {}

    # Get parameter from previously loaded dictionary, otherwise prompt user for input
    try:
        value = p_dict[variable]
        # if only part of the parameter information has been gathered from the user
        try:
            if (math.isnan(value) or (value == np.nan)) and (ind.ix[variable].isnull().sum()==1):
                no_info_in_file = True
            else:
                no_info_in_file = False
        except TypeError:
            no_info_in_file = False
    except:
        no_info_in_file = True
    
    if no_info_in_file:
        print("I cannot find the " + parameter + " for variable '" + 
              variable + "'. Please enter is here:")
        value = input()
        confirm = 'n'
        while confirm.lower() != 'y':
            print("You have assigned '" + value + "' as the " + parameter +
                  " for '" + variable + "'.")
            print("Is this correct? This will be saved for future use. Type 'y'/'n'")
            confirm = input()
            if confirm.lower() == "n":
                print("Please input the correct " + parameter + 
                      " for '" + variable +"':")
                value = input()
    
        # Add new variable to parameter dictionary
        p_dict[variable]=value
        
        # Save to file
        
        # Get new parameter array
        df1 = pd.DataFrame.from_dict(p_dict, orient='index')
        df1.index.name='variable'
        df1.columns = [parameter]
        # Get old parameter array
        df2 = ind[[[x for x in ['units','long_name'] if x != parameter][0]]]
        # Join the two
        df = df1.join(df2)
        # ensure order of columns remains 
        df = df[['units','long_name']] 
        df.index.name='variable'
        df = df.reindex(sorted(df.index, key=lambda x: x.lower()))
        df.to_csv('variable_description.csv')
        
        print("'" + value +"' has been assigned and saved as the " + 
              parameter + " for '" + variable + "'")
    else:
        print("Retrieved " + parameter + " for " + variable + " from file.")
        
    return value

def hdf_to_netcdf(h5_filename,
                 h5_key,
                 h5_dir = None,
                                  
                 global_title = None,
                 global_description = None,
                 author = 'Ruhi Humphries',
                 global_institution = 'CSIRO',
                 global_comment = None
                 
                 ):
    '''
    Converts a h5 file to a netcdf file
    '''
    if h5_dir is not None:
        os.chdir(h5_dir)
    # Load data from file
    df = pd.read_hdf(h5_filename,key=h5_key)
        
    # Save to netcdf
    df_to_netcdf(df, h5_filename, h5_dir,
                 global_title = global_title,
                 global_description = global_description,
                 author = author,
                 global_institution = global_institution,
                 global_comment = global_comment)
    
    return

def df_to_netcdf(df,
                 nc_filename,
                 nc_dir = None,
                 
                 global_title = None,
                 global_description = None,
                 author = 'Ruhi Humphries',
                 global_institution = 'CSIRO',
                 global_comment = None
                 ):
    '''
    Saves a dataframe to netcdf format
    '''
    # Setup all the inputs before writing to file
    fname = nc_filename.split('.')[0] + '.nc'
    if type(df) == pd.core.frame.Series:
        df = df.to_frame()
    if nc_dir is not None:
        os.chdir(nc_dir)

    if global_title == None:
        print('Please input a title (global attribute) of your dataset:')
        global_title = input()
    if global_description == None:
        print('Please input a description (global attribute) of your dataset:')
        global_description = input()
    if global_institution == None:
        print('Please input a institution (global attribute) of your dataset:')
        global_institution= input()
    if global_comment == None:
        print('Please input a comment (global attribute) of your dataset:')
        global_comment = input()
    
    print("Creating netCDF file: " + fname)
    # Open a new NetCDF file in write ('w') mode
    w_nc = nc.Dataset(fname,'w', format='NETCDF4')

    # Create global attributes
    w_nc.description = global_description
    w_nc.history = "Created at " + time.ctime(time.time()) + " by " + author
    w_nc.title = global_title
    w_nc.institution = global_institution
    w_nc.comment = global_comment
 
    # Create a set of dimensions
    w_nc.createDimension('time',None)

    # Create time variable
    print("Creating time variable")
    times = w_nc.createVariable('time',np.float64,('time',))
    times.units = 'hours since 2000-01-01 00:00:00'
    times.calendar = 'gregorian'
    times[:] = nc.date2num(df.index.tolist(),times.units,times.calendar)
    
    # Create other data variables
    var_dict = {} #Initialise
    for var in df.columns:
        vtype = df[var].dtype
        try:
            var_dict[var] = w_nc.createVariable(var,vtype,('time',))
        except TypeError:
            if vtype == bool:
                vtype = 'u1'
            var_dict[var] = w_nc.createVariable(var,vtype,('time',))
            
        # Create attributes for the variables
        var_dict[var].units = find_variable_parameter(var,'units')
        var_dict[var].long_name = find_variable_parameter(var,'long_name')
        var_dict[var].fill_value = np.nan
        
        #update user
        print("Saving " + var + " to netCDF file")
        
        # Add data to variables
        var_dict[var][:] = df[var].as_matrix()
        
    # close the new file
    w_nc.close()  
    
    cwd = os.getcwd()
    print('Successfully saved data as netcdf file in: ' + cwd + '\\' + fname)
    return

def write_netcdf(data,
                 # Variable attributes
                 standard_name, # the code-friendly name of the variable
                 long_name,
                 units,
                 
                 # File specification                 
                 filename,
                 path = '',
                 
                 type_var = None,
                 # Optional variable attributes - a subsample is included here. For full list, see http://salishsea-meopar-tools.readthedocs.io/en/latest/netcdf4/
                 valid_min = None,
                 valid_max = None,
                 
                 #Global attributes
                 g_Conventions = None,
                 g_title = 'Dataset title',
                 g_institution = 'CSIRO, Australia',
                 g_source = None,
                 g_history = None,
                 g_comment = None                 
                 ):


    '''
    Data type options:
    Type	Description
    f4	32-bit floating point
    f8	64bit-signed integer
    i1	8-bit signed integer
    i2	16-bit signed integer
    i4	32-bit signed integer
    i8	64-bit signed integer
    u1	8-bit unsigned integer
    u2	16-bit unsigned integer
    u4	32-bit unsigned integer
    u8	64-bit unsigned integer
    S1	Single-character string
    f 	Equivalent to f4
    d 	Equivalent to f8
    h 	Equivalent to i2
    s 	Equivalent to i2
    b 	Equivalent to i1
    B 	Equivalent to i1
    c 	Equivalent to S1
    i 	Equivalent to i4
    l 	Equivalent to i4

    '''

    # Check if file already exists, if so, open to append, if not, create new
    if os.path.isfile(filename):
        #  append mode
        nc_file = nc.Dataset(filename,'r+')
        Create = False
    else:
        # Create a new file ready for writing to 
        # For format, you can choose from 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'. Default is NETCDF4
        nc_file = nc.Dataset(filename,'w')
        Create = True
        
    #nc_file.description(description_str)
    
    if Create:
        # Create a dimension, for time, set the length of the dimension to unlimited for future appending
        nc_file.createDimension('time', None)
        # Create global attributes
        if g_Conventions is not None:
            nc_file.Conventions = g_Conventions
        nc_file.title = g_title
        nc_file.institution = g_institution
        if g_source is not None:        
            nc_file.source = g_source
        if g_history is not None:
            nc_file.history = g_history
        if g_comment is not None:
            nc_file.comment = g_comment
    
    # Check if the data is a timestamp, if so, calculate the number of seconds since a date, which is how NetCDF stores time data    
    if type(data) is pd.tseries.index.DatetimeIndex:
        timediff = data - datetime.datetime(2000,1,1,0,0,0)
        epoch_secs = timediff.seconds + timediff.days*24*60*60
        data = epoch_secs        
        units = 'seconds since 2000-1-1 00:00:00'
        long_name = 'time'
    else:
        # Otherwise, convert pandas series to a numpy array as expected by NetCDF
        data = data.as_matrix() 

    # Convert any non code friendly names
    '_'.join(standard_name.split(sep=' '))   # Check if there are spaces
    '_'.join(standard_name.split(sep='-'))   # Check if there are dashes
    
    if type_var is None:
        type_var = data.dtype
    # Create new variable   
    test_var = nc_file.createVariable(standard_name,type_var,('time'), zlib=True)
    
    # Create variable attributes
    test_var.units = units
    test_var.long_name = long_name
    test_var.standard_name = standard_name
    test_var.fill_value = np.nan
    # Set valid values. min and max should only be used if only one of them are specified. If both are, valid_range is used
    if (valid_max is not None) and (valid_min is not None):
        test_var.valid_range = np.array((valid_min,valid_max))
    elif valid_max is not None:
        test_var.valid_max = valid_max
    elif valid_min is not None:
        test_var.valid_min = valid_min
    
    
    # Write data
    test_var[:] = data
    
    
    nc_file.close()

    return
    
#Because acsm 10 min data doesn't align to a continuous 10 min period (e.g. 8:10 sometimes, 9:01 others, etc)
# we're going to average the 5 second data into 10 minute data, but utilising the start time of the acsm data.
def variable_timebase_resample(data,
                               desired_timebase,
                               desired_interval = '10Min'
                               ):
    import re

    # interpret the time interval and convert to a timedelta object
    desired_interval = desired_interval.replace(" ","") # Remove any whitespace in input
    time_int = re.split('(\d+)',desired_interval) # split into the number and the unit
    
    if (time_int[2].lower() == 'min') | (time_int[2].lower() == 'm') | (time_int[2].lower() == 'minute') | (time_int[2].lower() == 'minutes'):
        td = pd.Timedelta(minutes = float(time_int[1]))
    elif (time_int[2].lower() == 'sec') | (time_int[2].lower() == 's') | (time_int[2].lower() == 'second') | (time_int[2].lower() == 'seconds'):
        td = pd.Timedelta(seconds = float(time_int[1]))
    elif (time_int[2].lower() == 'hr') | (time_int[2].lower() == 'h')| (time_int[2].lower() == 'hour') | (time_int[2].lower() == 'hours'):
        td = pd.Timedelta(seconds = float(time_int[1]))
    else:
        print("Can't interpret your desired_interval units")
        return
        
    # Create new dataframe with desired timebase as index
    data_resamp = pd.DataFrame(index = desired_timebase, columns = data.columns)
    
    for i in range(1,len(desired_timebase)):
        # Find the interval to get stats over
        interval = (data.index > desired_timebase[i]) & (data.index < desired_timebase[i]+td)        
        data_resamp.iloc[i] = data[interval].mean()


    return data_resamp    
    
def read_filelist_from_file(filelist_filename = 'files_loaded.txt'):
    import pickle
    try:
        with open(filelist_filename, 'rb') as f:
            files_already_loaded = pickle.load(f) 
        return files_already_loaded
    except:
        return ['']
    
    
def write_filelist_to_file(filelist, filelist_filename = 'files_loaded.txt'):
    import pickle
    #Save the filenames that have been loaded to file for next update
    with open(filelist_filename, 'wb') as f:
        pickle.dump(filelist, f)
    return
#%%   
'''
import sys
sys.path.append('c:\\Dropbox\\RuhiFiles\\Research\\ProgramFiles\\pythonfiles\\')

import os
import matplotlib.pyplot as plt
import numpy as np
np.seterr(invalid='ignore')
import pandas as pd
import RVI_Underway
import CPC_TSI
import CCNC
import CAPRICORN
import Filter_Timeseries as fTS
import atmosplots

### Plotting maps
# import cartopy
# import basemap
# import folium

#%% CAPRICORN MASTER PROCESSING DOCUMENT
MASTER_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Processing/Aerosols/'
uwy_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/Underway/'
CCN_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/CCNC/'
CN3_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/TSI_SMPSCPC/3776_CN3/'
CN10_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/TSI_SMPSCPC/3776_CN10/'
smpsgrimm_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/SMPS_GRIMM'    
    
#%% Create datasets for each level of processing
os.chdir(MASTER_path)
cap = pd.read_hdf('aerosol_merged.h5',key='aerosol')
manual_exhaust = CAPRICORN.exhaust_filt()

RVI_Underway.create_exhaust_mask(cap, 
                                 mask_level_num = 1, 
                                 Filter4WindDir = True, 
                                 Filter4BC = True,
                                 Filter4O3 = True,
                                 Filter4CNstd = True,
                                 WD_exhaust_upper_limit = 277, WD_exhaust_lower_limit = 97,
                                 BC_lim = 0.05,
                                 CN_std_ID = 'CN3', CN3_std_lim = 150, 
                                 manual_exhaust_mask = manual_exhaust
                                 )
capL1 = cap.copy()
capL1.loc[cap['exhaust_mask_L1'].isnull()] = np.nan

#%%
if os.path.isfile('test.nc'):
    os.remove('test.nc')
write_netcdf(capL1.index,
             standard_name = 'time')
write_netcdf(capL1['ccn_med'],
             standard_name = 'ccn_med',
             units = 'number conc per cm3')
'''