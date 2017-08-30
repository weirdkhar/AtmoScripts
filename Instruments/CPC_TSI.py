"""
Function related to the loading and processing of CPC instruments from TSI

version: 0.0
date: 2016-09-09
"""
import sys
sys.path.append('h:\\code\\')
import pandas as pd
import os
import glob
import pickle
import numpy as np
import re
from AtmoScripts import atmoscripts
import matplotlib.pyplot as plt


def Load_to_HDF(input_path= None,
                input_filelist = None,
                output_path = None,
                output_h5_filename = 'CPC_sec', 
                InputTZ = 0, 
                OutputTZ = 0, 
                resample_time = False, 
                output_file_frequency = 'all',
                force_reload_from_source = False
                ):
    """
    Load performed after data has been exported to CSV file with just raw concentrations and times.
    """
    if output_path is None:
        os.chdir(input_path)
    else:
        os.chdir(output_path)
    
    if (output_h5_filename is None) or (output_h5_filename == ''):
        output_h5_filename = 'CPC'
        
    output_h5_filename = output_h5_filename + '_raw'
        
    if force_reload_from_source:
        remove_previous_output('h5',force_reload_from_source, input_filelist)
    
    if input_filelist is None:
        os.chdir(input_path)
        filelist = glob.glob('*.csv')
        # Check if previous data has been loaded, if so, don't load it again
        if os.path.isfile('files_loaded.txt'):
            with open('files_loaded.txt', 'rb') as f:
                files_already_loaded = pickle.load(f)
            # Get only the new files to be loaded:
            filelist=list(set(filelist).difference(set(files_already_loaded)))
        # Read where the import has gotten up to previously:
            if os.path.isfile('partial_files_loaded.txt'):
                with open('partial_files_loaded.txt','r') as f:
                    last_loaded = f.readlines()
                    last_loaded = [x.strip() for x in last_loaded]
                    last_loaded_file = last_loaded[0]
                if type(last_loaded_file) is str:
                    filelist.append(last_loaded_file)
    else:
        filelist = input_filelist
    filelist.sort()     

    #Iterate through to load the raw files 
    for file in filelist:
        # Read cpc csv file
        read_cpc_csv(file, output_h5_filename, output_file_frequency, 
                     InputTZ, OutputTZ)  
    
    # Clean up
    if os.path.isfile('partial_files_loaded.txt'):
        os.remove('partial_files_loaded.txt')
        
    #Save the files that have already been loaded to file for next update
    with open('files_loaded.txt', 'wb') as f:
        try:
            files_already_loaded
        except NameError:
            filelist = filelist
        else:
            filelist = filelist + files_already_loaded
        pickle.dump(filelist, f)

    if resample_time:    
        timebase_resampler(input_path, output_h5_filename,
                           variable = output_h5_filename,
                           time_int=['5S'],
                           output_path = output_path)
    
    return output_h5_filename 

def Load_to_NonHDF(input_path= None,
                input_filelist = None,
                output_path = None,
                output_h5_filename = 'CPC_sec', 
                InputTZ = 0, 
                OutputTZ = 0, 
                resample_time = False, 
                output_file_frequency = 'all',
                force_reload_from_source = False,
                output_file_format = 'csv',
                
                gui_mode = True,
                gui_mainloop = None
                ):
    ''' 
    Do all the processing in HDF, then save the final product as netcdf or csv
    '''
    assert output_file_format in ['csv','netcdf','nc'],'Choose either netcdf \
                                                        or csv file format!'
    
    # Load the data quickly via hdf
    base_fname = Load_to_HDF(
                input_path= input_path,
                input_filelist = input_filelist,
                output_path = output_path,
                output_h5_filename = output_h5_filename, 
                InputTZ = InputTZ, 
                OutputTZ = OutputTZ, 
                resample_time = resample_time, 
                output_file_frequency = output_file_frequency,
                force_reload_from_source = force_reload_from_source
                )
    
    os.chdir(output_path)
    # Get the list of recently created hdf files
    filelist = glob.glob('*'+base_fname+'*.h5')
    
    # Load each file then save it as the requested file formatr
    for f in filelist:
        d = pd.read_hdf(f,key = 'cn')
        if output_file_format.lower() == 'csv':
            fname = f.split('.')[0]+'.csv'
            d.to_csv(fname)
        else:
            fname = f.split('.')[0]+'.nc'
            atmoscripts.df_to_netcdf(
                    d,
                    nc_filename = fname,
                     
                    global_title = None,
                    global_description = None,
                    author = None,
                    global_institution = None,
                    global_comment = None,
                    
                    gui_mode=gui_mode,
                    gui_mainloop = gui_mainloop
                    )
    
    os.chdir(output_path) 
    if os.path.isfile('netcdf_global_attributes.temp'):
        os.remove('netcdf_global_attributes.temp')
    
    return


def load_cn(data_path=None, filetype=None, fname=None):
    ''' 
    Loads data from concatenated data file.
    '''
    if fname is None:
        assert data_path is not None, 'specify data path!'
        assert filetype is not None, 'specify filetype!'
    
        os.chdir(data_path)
        # Get most recently updated file:
        filelist = glob.glob('*.'+filetype)
        fname = min(filelist, key=os.path.getctime)
    else:
        filetype = fname.split('.')[-1]
        
    if filetype in ['hdf','h5']:
        data = pd.read_hdf(fname, key='cn')
    
    elif filetype in ['netcdf','nc']:
        data = atmoscripts.read_netcdf(fname, data_path)
    
    elif filetype == 'csv':
        data = pd.read_csv(fname,skipinitialspace = True,
                           index_col=0, 
                           parse_dates=True,
                           infer_datetime_format=True)  
    
    return data


def remove_previous_output(filetype, reload_from_source, input_flist):
    '''
    Checks if previous files have been created. If not, then return true and 
    create the new files. If so, and you've been asked to reload_from_source,
    return true. Otherwise, return false and don't reload the files. 
    '''
    input_filelist = [f.split('/')[-1] for f in input_flist]
    filelist = glob.glob('*'+filetype)
    if len(filelist) > 0 and reload_from_source:
        # Delete files and return
        for file in filelist:
            if file not in input_filelist:
                os.remove(file)
        if os.path.isfile('files_loaded.txt'):
            os.remove('files_loaded.txt')
    return

def save_to_hdf(data, output_h5_filename, output_file_frequency):    
    import datetime
    ''' Determine the destination file for each datapoint in the dataframe'''
    year_str = [str(i.isocalendar()[0]) for i in data.index]
    mnth_str = ['0' + str(i.month) \
                if i.month <10 \
                else str(i.month) \
                for i in data.index]
    
    
    if output_file_frequency.lower() == 'monthly':
#        print('Saving to monthly HDF files')
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+y for x,y in zip(year_str,mnth_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])
    
    elif output_file_frequency.lower() == 'weekly':
 #       print('Saving to weekly HDF files')
        
        wk_str = ['0'+str(i.isocalendar()[1]) \
                          if i.isocalendar()[1] < 10 \
                          else str(i.isocalendar()[1]) \
                          for i in data.index]
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+'_wk'+y for x,y in zip(year_str,wk_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])
        
    elif output_file_frequency.lower() == 'all':
        # Continue as normal
#        print('Saving all data to a single HDF file')
        data['destination_file'] = output_h5_filename
        output_filelist = set(data['destination_file'])
    else:
#        if output_file_frequency.lower() == 'daily':
#            print('Saving to daily HDF files')
#        else:
        if output_file_frequency.lower() != 'daily':
            print("Cannot determine what frequency you want the output file, defaulting to saving to daily files.")
        
        day_str = [datetime.datetime.strftime(i, format = '%d') for i in data.index]
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+y+z for x,y,z in zip(year_str,mnth_str,day_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])  
    
    
    ''' Save the appropriate data to each destination file '''
    for file in list(output_filelist):
        # Select data
        data_temp = data[data['destination_file']==file]
        # Delete extraneous columns
        del data_temp['destination_file']
        
        # Check if file already exists, if so, append, otherwise, create a new file
        if os.path.isfile(file+'.h5'):
            data_saved = pd.read_hdf(file+'.h5', key = 'cn')
            data_temp = data_saved.append(data_temp)
            # Drop any duplicates which may be there, based only on the Timestamp
            data_temp = data_temp.reset_index().drop_duplicates(subset='Timestamp', keep='last')
            data_temp = data_temp.set_index('Timestamp')
            
            #os.remove(file+'.h5')
        # Save to file
        data_temp.to_hdf(file+'.h5', key = 'cn', mode='a')
        
    
    # Remove additional columns that were added to the dataframe in the processing
    del data['destination_file']
    
    return file+'.h5'

def check_cpc_file_format(filename):
    '''
    Reformat the CPC output that is produced by AIM auto-export. 
    This involves removing header information which is reprinted after each 
    sample
    '''
    # read the file
    with open(filename) as f:
        # Read the file
        content = f.readlines()
        content = [x.strip() for x in content]
    
    # keep the first header so the read_cpc_csv function still works
    first_header = True 
    # Initialise new list
    content_reformatted = []
    header_list = []
    # Iterate through each line, each for validity       
    for line in content:
        if first_header and (line.split(',')[0] in ['Sample File','Model','','Sample #']):
            append_to_list(content_reformatted,line)
            continue
        elif line.split(',')[0] in ['Sample #']:
            append_to_list(header_list,line)
        else:
            try:
                int(line.split(',')[0])
                append_to_list(content_reformatted,line)
                first_header=False
            except:
                continue
      
    
    if 'Sample File' not in content[0]:
        return None
    
    if len(content) == len(content_reformatted):
        return filename
    else:
        # Make sure the header reflects the longest sample length in the file
        content_reformatted[3] = max(header_list,key=len)
        # Write new file
        filename_reformatted = filename.split('.')[0]+'_reformatted.'+filename.split('.')[1]
        if os.path.isfile(filename_reformatted):
            os.remove(filename_reformatted)
        with open(filename_reformatted,'wt') as fnew:
            fnew.write('\n'.join(line for line in content_reformatted))
        print('Input CPC file reformatted')
        # Return new filename 
        return filename_reformatted

def append_to_list(lst, line):
    return lst.append(line)    

def read_cpc_csv(read_filename, output_filename_base, output_file_frequency, InputTZ=0, OutputTZ=0):
    '''
    Reads CPC data exports from AIM 10 and higher as row based, with 
    ONLY concentration data output
    '''
    import numpy as np
    
    if output_file_frequency == 'all':
        print('Saving all data to a single HDF file')
    else:
        print('Saving to ' + output_file_frequency + ' HDF file')
    
    # Check format of file:
    read_filename = check_cpc_file_format(read_filename)
    if read_filename is None:
        return
    # Read each row of data, taking into account that each row can change length and parsing format (weird...)
    df = pd.read_csv(read_filename, skiprows = range(0,3), engine='python', skipinitialspace=True, iterator = True, chunksize = 1000)
    
    # Read the number of samples in the file
    with open(read_filename) as f: 
        lastline = f.readlines()[-1]
    numsamples = lastline.split(",")[0]
    
    # Read where the import has gotten up to previously:
    first_load = True
    if os.path.isfile('partial_files_loaded.txt'):
        with open('partial_files_loaded.txt','r') as f:
            last_loaded = f.readlines()
            last_loaded = [x.strip() for x in last_loaded]
            last_loaded_file = last_loaded[0]
            last_loaded_sample = int(last_loaded[1])
            first_load = False
    
    for chunk in df:
        # Extract initial timestamp for each sample (i.e. each row)
        try:
            chunk['sample_timestamp'] = pd.to_datetime(chunk['Start Date']+' '+chunk['Start Time'], format = '%m/%d/%y %H:%M:%S')
            chunk = chunk.reset_index()
            del chunk['index']
        except KeyError:
            # The csv file that you've read isn't actually a TSI CPC file
            return
        
        data = pd.DataFrame(columns = {'Timestamp', 'Concentration'})
        for rowidx in range(0,len(chunk)):
            if not first_load:
                if (chunk['Sample #'][rowidx] <= last_loaded_sample) and (read_filename == last_loaded_file):
                    continue
            # Create timestamp and extract concentration for each sample in chunk
            timestamp = [chunk['sample_timestamp'][rowidx]+pd.Timedelta(seconds=x) for x in range(0,chunk['Sample Length'][rowidx])]            
            conc = chunk.loc[rowidx][12:(12+chunk['Sample Length'][rowidx])]
            
            print('Formatting sample ' + str(chunk['Sample #'].loc[rowidx]) 
                  + ' of ' + numsamples + ' from file ' + read_filename)
            
            # Format data as dataframe
            data_temp = pd.DataFrame({'Timestamp': timestamp, 'Concentration': conc.values})
            # Append new data to current data
            data = pd.concat([data,data_temp])
      
        if len(data) != 0:
            # Drop duplicates that may be present
            data = data.drop_duplicates(subset='Timestamp', keep='last')
            # Set index
            data = data.set_index('Timestamp')
            # Coerce data to the correct type, dealing with infinite values output from AIM
            data['Concentration'] = [np.nan if x == '1.#INF'  else float(x) for x in data['Concentration']]
            
            #Correct for Timezone offsets caused by AIM exporting process
            if InputTZ-OutputTZ != 0 :
                data = TimeZoneCorrection(data, CurrentTZ = InputTZ, OutputTZ = OutputTZ)
            
            # Save to hdf file
            print('Saving chunk to file')
            outputfilename = save_to_hdf(data, output_filename_base, output_file_frequency)
    
            # Alert the user where the process is up to
            print('Data loaded from ' + read_filename +' and samples ' + str(chunk['Sample #'].iloc[0]) 
                        + ' to ' + str(chunk['Sample #'].iloc[-1]) + ' saved to ' + outputfilename)
    
    if os.path.isfile('partial_files_loaded.txt'):
        os.remove('partial_files_loaded.txt')
    with open('partial_files_loaded.txt','wt') as f:
        f.write(read_filename)
        f.write('\n')
        f.write(numsamples)
    
    # Make sure the file is closed
    f = open(read_filename,'r')
    f.close()
    return
    
def read_cpc_csv_row_AIM9(read_filename, output_filename_base, output_file_frequency, InputTZ=0, OutputTZ=0):
	# function to read row based cpc exported data from AIM versions 9 or below. 
	# This is VERY slow due to the exported format from AIM 
	#(error messages from sampling accumulate with each sample, causing the parsing of the data to slow down as it progresses).
	
    import datetime
    
    if output_file_frequency == 'all':
        print('Saving all data to a single HDF file')
    else:
        print('Saving to ' + output_file_frequency + ' HDF file')
    
    data_present = True
    i = 0
    while data_present:
        
        # Read each row of data, taking into account that each row can change length and parsing format (weird...)
        rows = pd.read_csv(read_filename, skiprows = range(0,3+i), engine='python', skipinitialspace=True, nrows=1)
        
        if len(rows)>0: # haven't yet reached the end of the file   
            # The reading of the rows is random, so need a few different techniques for reading
            if type(rows.index) != pd.indexes.multi.MultiIndex:
                # Extract initial timestamp
                sample_timestamp = pd.to_datetime(rows.iloc[0,0]+' '+rows.iloc[0,1], format = '%m/%d/%y %H:%M:%S')
                # Extract sample length
                sample_length = int(rows.iloc[0,2])
                # Extract sample number
                samplenum = rows.index[0]
            
            elif len(rows.index[0]) == 2:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.iloc[0,0], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.iloc[0,1])
                samplenum = rows.index[0][0]
            elif len(rows.index[0]) == 3:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.index[0][2], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.iloc[0,0])
                samplenum = rows.index[0][0]
            elif len(rows.index[0]) >= 4:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.index[0][2], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.index[0][3])
                samplenum = rows.index[0][0]
      
            # Make sample timestamp array
            timestamps = [sample_timestamp + datetime.timedelta(seconds=x) for x in range(0, sample_length)]
            # Remove all non-numeric data (e.g. errors)
            data_num = rows.select_dtypes(include=['number'])
            # Remove all additional data that is output in the file
            conc = data_num.iloc[:,8:(data_num.shape[1]-1)]
            # Remove any nans
            conc = conc.dropna(axis=1)
                        
            
            # Create dataframe with the extracted data
            sample_data = {'Timestamp' : pd.Series(timestamps), 'Concentration' : pd.Series(conc.values[0,:])}
            sample_data = pd.DataFrame(sample_data)
            sample_data = sample_data.set_index('Timestamp')
    
            #Correct for Timezone offsets caused by AIM exporting process
            if InputTZ-OutputTZ != 0 :
                sample_data = TimeZoneCorrection(sample_data, CurrentTZ = InputTZ, OutputTZ = OutputTZ)
            
            save_to_hdf(sample_data, output_filename_base, output_file_frequency)
            
            print('Successfully saved sample ' + str(samplenum) + ' to file.')
            
            i += 1 #iterate
      
        else:
            data_present=False       

    return

def read_cpc_csv_column(read_filename, output_filename_base, output_file_frequency):
    # function to read column based cpc exported data from AIM versions 9 or below. 
	# This is slow - preferencially use the row based function 
	
    colnames = ['Timestamp','Concentration'] #Instantiate    
    
    # Load date
    date_row = pd.read_csv(read_filename,
                           nrows = 1,
                           skiprows = range(0,4)
                           )
    
    # Find the number of samples in the file
    temp = pd.read_csv(read_filename,nrows = 1,skiprows = range(0,2))
    j_lim = ''
    k = 1
    while isinstance(j_lim,str) & (k > 0):
        if 'Unnamed' not in temp.columns[-k]:
            j_lim = int(temp.columns[-k])
        k = k+1
    
    del temp

    #Iterate through each sample            
    for j in range(1, j_lim+1): 
    
        date_str = date_row.columns[2*j-1].split('.')[0]
        #date = pd.to_datetime(date_str)        
        
        
        data_temp = pd.read_csv(read_filename, 
                                names = colnames, 
                                skiprows = range(0,18),
                                nrows = 298, # ignore comments at the end of the file
                                engine='python',
                                skipinitialspace = True, 
                                iterator=True,
                                usecols=[2*j-2,2*j-1]#range(2*j-2,2*j)
                                )
        
        # Get the index of the last data point in the column, i.e. ignoring the comments at the end of the file
        #k_end = data_temp[data_temp.Concentration.isnull()].index[0]
        #data_temp = data_temp[0:k_end]
        
        data_temp['Timestamp'] = pd.to_datetime(date_str+ ' ' + data_temp['Timestamp'])
        
        # Increment the date when the time moves over midnight
        for i in range(0, len(data_temp)):
            if data_temp['Timestamp'][i] < data_temp['Timestamp'][0]:
                data_temp['Timestamp'][i] = data_temp['Timestamp']+pd.Timedelta(days=1)

        #Set index to Timestamp
        data_temp = data_temp.set_index('Timestamp') 
        
        save_to_hdf(data_temp, output_filename_base, output_file_frequency)
        
        print('Successfully saved sample ' + str(j) + ' to file.')
        
        del data_temp

    
    return #data

def TimeZoneCorrection(DataFrame, CurrentTZ, ConvertToUTC = True, OutputTZ = 0):
    import pandas as pd
    
    if ConvertToUTC:
        OutputTZ= 0
    elif OutputTZ == 0:
        print("You must define your output timezone if its not UTC")
        return        
    
    # Correct for any timezone offsest
    DataFrame = DataFrame.shift(freq=pd.Timedelta(hours=OutputTZ-CurrentTZ))
    
    return DataFrame


def flow_cal(data, 
             flow_cal_filename = None,
             flow_cal_path = None,
             measured_flows_df = None, 
             set_flow_rate = 1000, #ccm
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
         correct with. If polydeg is a string, the string will be input into the 
         "type" option of scipy.interpolate.interp1d . Options include:
             ‘linear’
             ‘nearest’, 
             ‘zero’ (zeroth order spline)
             ‘slinear’ (first order spline)
             ‘quadratic’ (second order spline) 
             ‘cubic’ (third order spline)
    '''
    
    # Load data from file if required:
    if measured_flows_df is None:
        # Assume we're already in the required directory if the path isn't 
        # provided
        measured_flows_df = load_flow_cals(flow_cal_filename, flow_cal_path)
    
    # Format index and sort it alphabetically
    measured_flows_df['Timestamp'] = pd.to_datetime(measured_flows_df.index)
    measured_flows_df = measured_flows_df.set_index('Timestamp')
    measured_flows_df = measured_flows_df.sort_index()
    
    # Convert dates to seconds since 1 Jan 2000
    x = (measured_flows_df.index - measured_flows_df.index[0]).total_seconds()
    y = measured_flows_df['flow rate']
    
    
    if type(polydeg) is str:
        if polydeg in ['linear', 'nearest', 'zero', 
                       'slinear', 'quadratic', 'cubic']:
            from scipy.interpolate import interp1d
            p = interp1d(x, y, kind=polydeg)
            # Abs uncertainty is the RMS deviation of the regression
            sigma_abs = 0
        else:
            print("Couldn't recognise polydeg string input, fitting second \
                  degree polynomial. Please check input")
            polydeg = 2
    if type(polydeg) is not str:
        fit = np.polyfit(x,y,deg=polydeg, full=True)
        p = np.poly1d(fit[0])
        # Abs uncertainty is the RMS deviation of the regression
        try:
            sigma_abs = fit[1][0]
        except IndexError: 
            # we get a perfect fit because we have < n calibration points
            sigma_abs = 0

    x_data = (pd.to_datetime(data.index) - \
              measured_flows_df.index[0]).total_seconds()

    data['Concentration'] = data['Concentration']/set_flow_rate*p(x_data)
   #plt.plot(x,y,'.',xp,p(xp),'--')
    
    
        
    # Rel uncertainty is abs divided by median of liniear regression
    sigma_divisor = p(x_data)
    data = uncertainty_calc(data,sigma_abs,sigma_divisor)
    
    return data
    
#==============================================================================
# def LoadAndProcess(CN_path, 
#                    filename_base = 'CN3', 
#                    filtOrRaw='filt', 
#                    timeResolution='',
#                    mask_period_timestamp_list = [''],
#                    CurrentTZ = 0, 
#                    OutputTZ = 0,
#                    CN_flow_check_df = '',
#                    CN_flow_setpt = 1500,
#                    CN_flow_polyDeg = 2
#                    ):
#     
#     os.chdir(CN_path)
#     NeedsTZCorrection=True #Initialise
#     
#     # Check if any h5 file has been produced yet (ie. the initial processing 
#     # has occurred)
#     if not glob.glob('*.h5'):
#         Load_to_HDF(CN_path,filename_base, InputTZ = CurrentTZ, OutputTZ = OutputTZ)
#     
#     filename = filename_base+'_'+timeResolution+'.h5'
#     filename_1sec = filename_base+'_'+filtOrRaw+'.h5'
# 
#     if filtOrRaw.lower() == 'filt':
#         if os.path.isfile(filename):
#             CN = pd.read_hdf(filename,key='cn')
#             NeedsTimeResampling = False
#             return CN
#         elif os.path.isfile(filename):
#             CN = pd.read_hdf(filename_1sec,key='cn')
#             NeedsTimeResampling = True   
#             return CN
#         else:
#             filtOrRaw = 'raw' # filt file not available, produce it!
#             filename_1sec = filename_base+'_'+filtOrRaw+'.h5'
#             
#     if filtOrRaw.lower() == 'raw':
#         if os.path.isfile(filename_1sec):
#             CN = pd.read_hdf(filename_1sec,key='cn')
#             NeedsTimeResampling = False
#             NeedsTZCorrection = False #Only correct for TZ when creating the H5 file for the first time. 
#         if os.path.isfile(filename): # Return the raw file if resampling has already been done
#             return CN
#         else:
#             NeedsTimeResampling = True
#     else:
#         print("No hdf file exists with the raw data! Please run the following function before this one: CPC_TSI.Load_to_HDF(RawDataPath,output_h5_filename = 'CPC', TZCorrect = False, InputTZ = 0, OutputTZ = 0,resample_time = False)")
#         return
#     
#     # Correct timezone if necessary
#     if NeedsTZCorrection:
#         if CurrentTZ - OutputTZ != 0:
#             if OutputTZ == 0:
#                 ToUTC = True
#             else:
#                 ToUTC = False
#             CN = TimeZoneCorrection(CN, CurrentTZ, ConvertToUTC = ToUTC, OutputTZ = 0)
# 
# 
#     NeedsFiltering = False #Initialise
#     if filtOrRaw.lower() == 'raw':
#         # Flow calibrations
#         if not isinstance(CN_flow_check_df,str):
#             CN = flow_cal(CN,CN_flow_check_df,CN_flow_setpt,polydeg=CN_flow_polyDeg)
#             NeedsFiltering = True
#         
#         # work through mask periods and set values to nan
#         for i in range(int(len(mask_period_timestamp_list)/2)):
#             CN.loc[(CN.index >= mask_period_timestamp_list[2*i]) & (CN.index < mask_period_timestamp_list[2*i+1])]= np.nan
#             NeedsFiltering = True
#         # Save to file as 1 second filtered data
#         if NeedsFiltering:
#             CN.to_hdf(filename_base+'_filt.h5',key = 'cn')
#     else:
#         print("Don't know what to load. Please specify either Raw or Filt")
#         return
# 
#     
#         
#     if NeedsTimeResampling & (not timeResolution == ''):     
#         # Check if current time resolution is what is being asked for, if not, resample. If so, floor to nearest interval
#         current_time_res = (CN.index[1] - CN.index[0]).seconds
#         if any(substring in timeResolution for substring in ['S', 'sec', 'Sec']):
#             if int(re.findall('\d+', timeResolution)[0]) == current_time_res:
#                 if 'Concentration' in CN:
#                     CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
#                 ns = 1*1*1000000000 # 1 second in nanoseconds
#                 CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns ))
#                 return CN
#         elif any(substring in timeResolution.lower() for substring in ['min']):
#             if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60:
#                 if 'Concentration' in CN:
#                     CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
#                 ns = 1*60*1000000000 # 1 minute in nanoseconds
#                 CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
#                 return CN
#         elif any(substring in timeResolution for substring in ['H', 'Hr', 'hr', 'Hour', 'hour']):
#             if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60/60:
#                 if 'Concentration' in CN:
#                     CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
#                 ns = 60*60*1000000000 #60 minutes in nanoseconds
#                 CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
#                 return CN
#         elif any(substring in timeResolution for substring in ['D', 'day', 'Day']):
#             if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60/60/24:
#                 if 'Concentration' in CN:
#                     CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
#                 ns = 24*60*60*1000000000 # 1 day in nanoseconds
#                 CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
#                 return CN
#         CN = resample_timebase(data = CN, variable = filename_base, time_int=[timeResolution])    
#         
#     return CN
#==============================================================================


    

def LoadAndProcess(cn_raw_path = None, 
                   cn_output_path = None,
                   cn_output_filetype = 'hdf',
                   load_from_filetype = 'csv',
                   filename_base = 'CN', 
                   force_reload_from_source = False,
                   output_time_resolution = '1S',
                   concat_file_frequency = 'all',
                   input_filelist = None,
                   
                   gui_mode = False,
                   gui_mainloop = None,
                   
                   NeedsTZCorrection = False,
                   CurrentTZ = 0, 
                   OutputTZ = 0,
                   
                   mask_period_file = None,
                   mask_period_timestamp_df = None,
                   
                   flow_cal_file = None,
                   flow_cal_df = None,
                   CN_flow_setpt = 1000,
                   CN_flow_polyDeg = 2,
                   
                   plot_each_step = False
                   ):
    '''
    Loads CPC data from csv files exported from AIM, concatenates, then saves 
    to output files of hdf, netcdf or csv format.
    Data can then be:
        - calibrated for flow rates 
        - correcting for time zone offsets resulting from timezone errors in 
        exporting data
        - filtering for logged events
        - If requested, it will perform exhaust removal (assuming its on the RVI)
    '''
    if cn_output_path is None:
        cn_output_path = cn_raw_path

    if load_from_filetype == "csv" and force_reload_from_source:
        load_data_to_file(
                          cn_raw_path = cn_raw_path, 
                          cn_output_path = cn_output_path,
                          filename_base = filename_base, 
                          cn_output_filetype = cn_output_filetype,
                          force_reload_from_source = force_reload_from_source,
                          input_filelist = input_filelist,
                          concat_file_frequency = concat_file_frequency,
                          InputTZ = CurrentTZ, 
                          OutputTZ= OutputTZ,
                          gui_mode = gui_mode,
                          gui_mainloop = gui_mainloop
                          )
    if (load_from_filetype == 'hdf') and (input_filelist is not None):
        raw_filelist = input_filelist
    else:
        raw_filelist = get_raw_filelist(cn_output_path,cn_output_filetype, 'raw')

    for file in raw_filelist:    
        # Load data
        try:
            data = load_cn(fname = file)
        except:
            if load_from_filetype == "csv":
                data = load_cn(cn_output_path,cn_output_filetype)
            else:
                data = load_cn(cn_raw_path, load_from_filetype)
        
        plot_me(data, plot_each_step,'Concentration','raw')
        
        
        # Correct timezone if necessary
        if NeedsTZCorrection:
            if CurrentTZ - OutputTZ != 0:
                if OutputTZ == 0:
                    ToUTC = True
                else:
                    ToUTC = False
                data = TimeZoneCorrection(data, 
                                        CurrentTZ, 
                                        ConvertToUTC = ToUTC, 
                                        OutputTZ = 0)
                
        # Calculate CN counting uncertainty
        data = uncertainty_calc(data,
                                    1,
                                    np.sqrt(data['Concentration']))
        
        # Perform flow calibration if data is provided
        if flow_cal_file is not None: 
            data = flow_cal(data,
                            flow_cal_file,
                            cn_raw_path, 
                            set_flow_rate=CN_flow_setpt,
                            polydeg=CN_flow_polyDeg
                            )
            save_as(data,cn_output_path,'flowCal',cn_output_filetype, file)
            plot_me(data, plot_each_step,'Concentration','flow cal')
        elif flow_cal_df is not None:
            data = flow_cal(data,
                            measured_flows_df=flow_cal_df,
                            set_flow_rate=CN_flow_setpt,
                            polydeg=CN_flow_polyDeg
                            )
            save_as(data,cn_output_path,'flowCal',cn_output_filetype, file)
            plot_me(data, plot_each_step,'Concentration','flow cal')
        
        # Correct for inlet losses #xkcd
    #    data = inlet_corrections(data, IE)
    #    save_as(data,cn_output_data_path,'IE',cn_output_filetype)
    #
    #   plot_me(data, plot_each_step,'CN Number Conc', 'IE')
        
        # Filter for logged events
        if mask_period_file is not None:
            data = atmoscripts.log_filter(data,
                                              cn_raw_path,mask_period_file)
            save_as(data,cn_output_path,'logFilt',cn_output_filetype, file)
            plot_me(data, plot_each_step,'Concentration','log filter')
        elif mask_period_timestamp_df is not None:
            data = atmoscripts.log_filter(data,
                                              log_mask_df=mask_period_timestamp_df)
            save_as(data,cn_output_path,'logFilt',cn_output_filetype, file)
            plot_me(data, plot_each_step,'Concentration','log filter')
            
            
        # Filter for exhaust #xkcd
        
    #    save_as(data,cn_output_path,'exhaustfilt',cn_output_filetype, file)
            
    
        # Resample timebase and calculate uncertainties
        data = timebase_resampler(data,time_int=output_time_resolution,
                                  input_h5_filename = file,
                                  output_filetype = cn_output_filetype,
                                  output_path = cn_output_path,
                                  gui_mode=gui_mode,
                                  gui_mainloop = gui_mainloop)
    
        if os.path.isfile('netcdf_global_attributes.temp'):
            os.remove('netcdf_global_attributes.temp')
    return data

def plot_me(data, plot_each_step, var=None, title = ''):
    if plot_each_step:
        if var is None:
            # Plot everything
            plt.plot(data)
        else:
            plt.plot(data[var])
        plt.title(title)
        plt.show()
    return

def get_raw_filelist(cn_output_path, output_filetype, substring='raw'):
    '''
    Retrieves a list of the raw files so that processing can be done on all 
    of them, not just the last one.
    '''
    os.chdir(cn_output_path)
    flist = glob.glob('*.'+output_filetype)
    raw_filelist = [f for f in flist if substring in f]
    raw_filelist.sort()
    return raw_filelist

def load_data_to_file(
                      cn_raw_path = None,
                      cn_output_path = None,
                      filename_base = 'cn',
                      cn_output_filetype = 'h5',
                      force_reload_from_source = False,
                      input_filelist = None,
                      concat_file_frequency = 'all',
                      InputTZ = 0, 
                      OutputTZ = 0,
                      gui_mode = True,
                      gui_mainloop = None
                      ):
    
    assert cn_output_filetype in ['hdf','h5','netcdf','nc','csv'], "Don't \
        recognise filetype to save to. Please use hdf, h5, netcdf or csv"
    
    if cn_output_filetype in ['hdf','h5']:
        filelist_empty = check_filelist('.h5', force_reload_from_source,input_filelist)
        if filelist_empty:
            Load_to_HDF(input_path = cn_raw_path,
                        output_path = cn_output_path,
                        output_h5_filename = filename_base,
                        output_file_frequency = concat_file_frequency,
                        input_filelist=input_filelist,
                        InputTZ = InputTZ, OutputTZ = OutputTZ,
                        force_reload_from_source = force_reload_from_source
                        )
    
    elif cn_output_filetype in ['netcdf','nc','csv']:
        filelist_empty = check_filelist('.'+cn_output_filetype, force_reload_from_source,input_filelist)
        if filelist_empty:
            Load_to_NonHDF(input_path = cn_raw_path,
                        output_path = cn_output_path,
                        output_h5_filename = filename_base,
                        output_file_frequency = concat_file_frequency,
                        input_filelist=input_filelist,
                        InputTZ = InputTZ, OutputTZ = OutputTZ,
                        force_reload_from_source = force_reload_from_source,
                        output_file_format = cn_output_filetype,
                        gui_mode = gui_mode,
                        gui_mainloop = gui_mainloop
                        )
    return


def check_filelist(filetype, reload_from_source,input_flist):
    '''
    Checks if previous files have been created. If not, then return true and 
    create the new files. If so, and you've been asked to reload_from_source,
    return true. Otherwise, return false and don't reload the files. 
    '''
    input_filelist = [f.split('/')[-1] for f in input_flist]
    filelist = glob.glob('*'+filetype)
    if len(filelist) > 0 and reload_from_source:
        # Delete files and return
        for file in filelist:
            if file not in input_filelist:
                os.remove(file)
        filelist_empty = True
    elif len(filelist) == 0:
        filelist_empty = True
        
    else:
        filelist_empty = False
    
    return filelist_empty

            
def save_as(data,
            save_path,
            filename_appendage = '',
            filetype='hdf',
            fname_current = None
            ):
    '''
    Saves data to file, reading the original filename, and appending informative
    text to the filename.
    For example
    CN.h5 becomes CN_QC.h5
    CN.nc becomes CN_QC_flowcal.nc
    '''
    assert filetype in ['hdf','h5','netcdf','nc','csv'], "Don't recognise \
                        filetype to save to. Please use hdf, h5, netcdf or csv"
    assert save_path is not None, 'You must specify the directory where you \
                        want to save!'
    os.chdir(save_path)
    
    
    if filetype in ['hdf','h5']:
        fname = get_filenamebase('h5', filename_appendage, fname_current)
        fname_components = fname.split('.')
        if fname_components[-1] == 'hdf':
            fname = fname_components[0]+'.h5'
        # Save data to file
        data.to_hdf(fname, key='cn')
    
    elif filetype in ['netcdf','nc']:
        # Get the filename of the most recently created file
        fname = get_filenamebase('nc', filename_appendage, fname_current)
        
        # Save data to file
        # xkcd
        
    elif filetype == 'csv':
        fname = get_filenamebase('csv', filename_appendage, fname_current)
        
        # Save data to file
        data.to_csv(fname)
        
    return

def get_filenamebase(ext, appendage, fname_current=None):
    '''
    Get's the filename of the most recently created file and produces the new
    filename
    '''
    
    filelist = glob.glob('*.'+ ext)
    
    if len(filelist) == 0:
        return 'CN_unknown' + appendage + '.' + ext
    else:
        if fname_current is not None:
            # Get the most recent version of the current file
            fname_current_base = fname_current.split('.')[0]
            fname_current_list = [f for f in filelist if fname_current_base in f]
            fname_old = max(fname_current_list,key=os.path.getctime)
            fname_old = fname_old.split('.')
        else:
            # Get the filename of the most recently created file
            fname_old = max(filelist, key=os.path.getctime).split('.')
            
        # if the version of the file is already there, overwrite
        if appendage in fname_old[0]:
            return fname_old[0] + '.' + fname_old[1]
        else:
            return fname_old[0] + '_' + appendage + '.' + fname_old[1]

#==============================================================================
# def resample_timebase(data=0,RawDataPath='',input_h5_filename='',variable='CN',time_int='default'):
#     ### Time resampling
# 
#     import pandas as pd
#     import numpy as np
#     import os
#     
#     if not isinstance(data, pd.DataFrame): #if no data provided, try to load from file
#         if (not RawDataPath == '') & (not input_h5_filename == ''):
#             os.chdir(RawDataPath)
#             if os.path.isfile(input_h5_filename): 
#                 data = pd.read_hdf(input_h5_filename+'.h5', key='cn')
#         else:
#             print("Please input either a dataframe or a datapath and filename where data can be found")
#             return
#     
#     
#     # define time resampling intervals unless specified in function input
#     if time_int == 'default':
#         time_int = ['5S','1Min', '5Min', '10Min', '30Min', '1H', '3H', '6H', '12H', '1D']    
#     
#     
#     # define MAD calculation
#     mad = lambda x: np.fabs(x - x.median()).median() 
#     
#     # define time resampling intervals
#     i_lim = len(time_int)
#     for i in range(0, i_lim):
#         t_int = time_int[i]    
#         # Initialise    
#         data_resamp = data['Concentration'].resample(t_int,fill_method=None, how=['count'])
#         data_resamp.rename(columns={'count' : variable+'_count'},inplace=True)
#         # Resample with stats    
#         data_resamp[variable+'_median'] = data.resample(t_int,fill_method=None).median()
#         data_resamp[variable+'_mad'] = data.resample(t_int,fill_method=None).apply(mad)
#         data_resamp[variable+'_mean'] = data.resample(t_int,fill_method=None).mean()
#         data_resamp[variable+'_std'] = data.resample(t_int,fill_method=None).std()
#         
#         #del data_resamp['Concentration']
#         
#         # Save to file
#         if isinstance(data,pd.DataFrame):        
#             outputfilename = variable+'_'+time_int[i]+'.h5'
#         else:
#             outputfilename = input_h5_filename+'_'+ time_int[i] +'.h5'
#         data_resamp.to_hdf(outputfilename, key='cn')
#     
#     return data_resamp
#==============================================================================
def timebase_resampler(
                      data=0,
                      RawDataPath='',
                      input_h5_filename='',
                      variable='cn',
                      time_int='default',
                      output_filetype = 'h5',
                      output_path = None,
                      gui_mode=False,
                      gui_mainloop = None
                      ):
    ''' 
    Time resampling
    '''
    #if no data provided, try to load from file
    if not isinstance(data, pd.DataFrame):
        if (not RawDataPath == '') & (not input_h5_filename == ''):
            os.chdir(RawDataPath)
            if os.path.isfile(input_h5_filename+'.h5'): 
                data = pd.read_hdf(input_h5_filename+'.h5', key='cn')
        else:
            print("Please input either a dataframe or \
                  a datapath and filename where data can be found")
            return
    
    
    # define time resampling intervals unless specified in function input
    if time_int == 'default':
        time_int = ['5S','1Min', '5Min', '10Min', \
                    '30Min', '1H', '3H', '6H', '12H', '1D']    
    elif type(time_int[0]) == bool:
        from itertools import compress
        gui_time_options = ['1S','5S','10S','15S','30S',
                             '1Min','5Min','10Min','15Min','30Min',
                             '1H','3H','6H','12H','1D']
        time_int = list(compress(gui_time_options, time_int))
    elif type(time_int) == str:
        time_int = [time_int]
    
    # define MAD calculation
    mad = lambda x: np.fabs(x - x.median()).median() 
    
    # define square root of the sum of squares calculation (root mean square numerator)
    rmsn = lambda x: np.sqrt(np.sum(x**2))
    
    for time in time_int:
        if time != '1S':
            data_temp = data.resample(time,fill_method=None).apply(rmsn)
            if 'cn_sigma' in data_temp:
                data_resamp = pd.DataFrame(data_temp['cn_sigma'])
                data_resamp.columns = ['cn_rmsn']
            else:
                data_resamp['cn_rmsn'] = 0 # if no processing has been done previously
            del data_temp
            for column in data.columns:
                if column != 'cn_sigma':    
                    sub_cn = pd.DataFrame(data[column].copy())
                    if column == 'Concentration':
                        prefix = 'cn'
                    else:
                        prefix = column
                    data_resamp[prefix+'_med'] = \
                        sub_cn.resample(time,fill_method=None).median()
                    data_resamp[prefix+'_mad'] = \
                        sub_cn.resample(time,fill_method=None).apply(mad)
                    data_resamp[prefix+'_avg'] = \
                        sub_cn.resample(time,fill_method=None).mean()
                    data_resamp[prefix+'_std'] = \
                        sub_cn.resample(time,fill_method=None).std()
                    data_resamp[prefix+'_count'] = \
                        sub_cn.resample(time,fill_method=None).count()
                    
            
            # Calculate uncertainty:
            data_resamp = uncertainty_calc_time_resample(
                                        data_resamp,
                                        'rmsn',
                                        'cn_count',
                                        'mad',
                                        col_name = 'cn_med',
                                        output_sigma_name='sigma_med'
                                                    )
            data_resamp = uncertainty_calc_time_resample(
                                        data_resamp,
                                        'rmsn',
                                        'cn_count',
                                        'std',
                                        col_name = 'cn_avg',
                                        output_sigma_name='sigma_avg'
                                                    )
            
            # Reorder columns based on name:
            data_resamp.sort_index(axis=1)
            
            # Remove temporary calculation
            del data_resamp['cn_rmsn']
            # Save to file
            save_resampled_data(data,data_resamp,time,
                                variable,input_h5_filename,
                                output_filetype,
                                output_path,
                                gui_mode=gui_mode,
                                gui_mainloop = gui_mainloop)
       
    try:
        
        return data_resamp
    except:
        return data
    
def uncertainty_calc_time_resample(data, 
                     abs_sigma, 
                     sigma_divisor,
                     dev_stat = 'std',
                     col_name = 'Concentration', 
                     output_sigma_name = 'cn_sigma'
                     ):
    '''
    Propagates measurement uncertainty and adds statistical uncertainty:
        
    '''
    # Find the rmsn column
    rmsn_cols = [col for col in data.columns if abs_sigma in col]
    # Find the count column
    count_cols = [col for col in data.columns if sigma_divisor in col]
    # Find the desired population deviation stat to use in sigma/sqrt(n)
    dev_cols = [col for col in data.columns if dev_stat in col]
    # Find the root names of each supersaturation
    ss_root_names = [col.split("_"+abs_sigma)[0] for col in rmsn_cols]
    
    # Sort columns
    rmsn_cols.sort()
    count_cols.sort()
    dev_cols.sort()
    ss_root_names.sort()
    
    for i in range(0, len(ss_root_names)):
        data[ss_root_names[i]+"_"+output_sigma_name] = \
                    np.sqrt(
                            (data[rmsn_cols[i]]/data[count_cols[i]])**2
                            +
                            (data[dev_cols[i]]/np.sqrt(data[count_cols[i]]))**2
                            )
    
    return data

            
def uncertainty_calc(data, 
                 abs_sigma, 
                 sigma_divisor,
                 col_name = 'Concentration', 
                 output_sigma_name = 'cn_sigma'
                 ):     
    '''
    Calculates and propogates uncertainty for each calibration process
    '''
    if 'cn_sigma' in data.columns:
        data[output_sigma_name] = data[col_name] * \
                            (
                            (abs_sigma/sigma_divisor)**2 
                            + 
                            (data['cn_sigma']/data[col_name])**2
                            )**0.5
    else: #Initialise
        data['cn_sigma'] = data['Concentration'] * \
                            ((abs_sigma/sigma_divisor)**2)**0.5
                            
    return data


def save_resampled_data(data, data_resamp,time_int,
                        variable = None, input_h5_filename = None,
                        output_filetype = 'h5',
                        output_path = None,
                        gui_mode=False,
                        gui_mainloop = None):
    if output_path is not None:
        os.chdir(output_path)
    if input_h5_filename is not None:
        s = input_h5_filename.split('.')
        outputfilename = s[0]+'_'+time_int+'.'+output_filetype
    elif isinstance(data,pd.DataFrame): 
        outputfilename = variable+'_'+time_int+'.'+output_filetype
    else:
        outputfilename = 'undefinedData_'+ time_int +'.'+output_filetype
    
    if output_filetype in ['h5','hdf']:
        fname_components = outputfilename.split('.')
        if fname_components[-1] == 'hdf':
            outputfilename = fname_components[0]+'.h5'
        data_resamp.to_hdf(outputfilename, key=variable)
    elif output_filetype in ['nc','netcdf']:
        atmoscripts.df_to_netcdf(data_resamp,outputfilename,
                                 gui_mode=gui_mode,
                                 gui_mainloop = gui_mainloop,
                                 nc_path = output_path)
    else:
        data_resamp.to_csv(outputfilename)
    
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
               
    
        if os.path.isfile(path+filename):
            file_FULLPATH = path+filename
        elif os.path.isfile(filename):
            file_FULLPATH = filename
        else:
            assert os.path.isfile(filename), filename + " could not be found"
    
    
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
    df = df.set_index(df.columns[0])
    df = df.take([0], axis=1)
    df.columns = ['flow rate']
    return df
