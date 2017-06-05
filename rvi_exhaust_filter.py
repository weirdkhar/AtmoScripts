'''
Script that takes a pandas dataframe of data from the RVI and adds an exhaust
column which is a boolean indicating True when measurements are affected by 
exhaust, and False when clean air is being sampled. 
The detection setup will identify other high intensity (and variable) sources,
but will leave diffuse urban influence.
'''
import sys
sys.path.append('h:\\code\\atmoscripts\\')
import os
import RVI_GHGs as ghg
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
from dateutil.parser import parse
import time
import ctypes  # An included library with Python install.
from scipy import signal
import matplotlib.pyplot as plt
import numba

local_path_cn = 's:\\in2016_v03\\cpc\\'
local_path_uwy = 's:\\in2016_v03\\uwy\\'
local_path_pic = 'r:\\RV_Investigator\\GHGs\\Picarro\\'
local_path_aer = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\'
local_path_ccn = 's:\\in2016_v03\\ccnc\\'

master_path = 'h:\\code\\AtmoScripts\\'

exhaust_path = 'r:\\RV_Investigator\\'


def main(exhaust_path = exhaust_path,
         startdate = '2016-05-06',
         enddate = '2016-05-07'
         ):
    df = create_exhaust_id(exhaust_path = exhaust_path,
                      startdate = '2016-05-06',
                      enddate = '2016-05-07'
                      )
    return df

def create_exhaust_id(df = None,
                      exhaust_path = exhaust_path,
                      startdate = '2016-05-06',
                      enddate = '2016-05-07',
                      force_reload = False,
                      force_recalculate = False,
                      abridged = True,
                      co_id = True,
                      cn_id = True,
                      bc_id = True,
                      filter_window = 60*10,
                      
                      co_stat_window = 10,
                      co_num_devs = 4,
                      #co_stat_threshold = 0.245,
                      
                      cn_stat_window = 10,
                      cn_num_devs = 4,
                      #cn_stat_threshold = 50,
                      
                      bc_lim = 0.07001
                      ):
    if df is None:
        if force_reload:
            delete_previous_files(exhaust_path)
        os.chdir(exhaust_path)
        e_flist = glob.glob('exhaust*.h5')    
    
        if (len(e_flist) > 0) and force_recalculate:
            df = pd.read_hdf(e_flist[0],key='exhaust')
            explode = e_flist[0].split('.')[0].split('_')
            voylst = [explode[2]+'_'+explode[3]]
            print('--- Loaded cn, co and bc data from previously created file')
        elif len(e_flist) > 0:
            df = pd.read_hdf(e_flist[0],key='exhaust')
            explode = e_flist[0].split('.')[0].split('_')
            voylst = [explode[2]+'_'+explode[3]]
            print('--- Loaded exhaust ID from previously created file:')
            print('    ' + exhaust_path + e_flist[0])
            return df
        else:
            df, voylst = load(startdate,enddate)
        

    print('--- Creating exhaust ID datastream from in-situ data...')
    df = ID_exhaust(df,
                    co_id = co_id,
                    cn_id = cn_id,
                    bc_id = bc_id,
                    filter_window = filter_window,
                    co_stat_window = co_stat_window,
                    co_num_devs = co_num_devs,                        
                    cn_stat_window = cn_stat_window,
                    cn_num_devs = cn_num_devs,                        
                    bc_lim = bc_lim)
    
    if abridged:
        df = df[['exhaust']]
    
    try: 
        voylst
    except:
        voylst = 'test'
    save_exhaust_id(df,exhaust_path,voylst)
    
    print('--- Exhaust ID created and files saved to ' + exhaust_path)
    return df

def delete_previous_files(path):
    os.chdir(path)
    flist = glob.glob('exhaust_id*')
    for file in flist:
        os.remove(file)
    return

def save_exhaust_id(df, 
                    path, 
                    voylst, 
                    saveh5 = True, 
                    savenc = True, 
                    savecsv = True):
    ''' Saves exhaust as h5, netcdf and csv file '''
    
    print('--- Saving to file')
    assert os.path.isdir(path), 'Path specified in saving function does not exist!'
    os.chdir(path)
    
    fname = 'exhaust_id'
    for voy in voylst:
        fname = fname+'_'+voy
        
    if saveh5:
        df.to_hdf(fname+'.h5',key='exhaust')
    
    if savenc:
        # Open a new NetCDF file in write ('w') mode
        w_nc = nc.Dataset(fname+'.nc','w', format='NETCDF4')

        # Create global attributes
        w_nc.description = "RV Investigator exhaust identification using BC, CN and CO data"
        w_nc.history = "Created " + time.ctime(time.time())
        w_nc.source = "Ruhi Humphries"
 
       # Create a set of dimensions
        w_nc.createDimension('time',None)

        # Create a variables
        times = w_nc.createVariable('time',np.float64,('time',))
        exhaust = w_nc.createVariable('exhaust', np.int8,('time',))

        # Create attributes for the variables
        times.units = 'hours since 2000-01-01 00:00:00'
        times.calendar = 'gregorian'
        exhaust.units = 'dimensionless'
        
        # Add data to variables
        exhaust[:] = df['exhaust'].as_matrix()
        times[:] = nc.date2num(df.index.tolist(),times.units,times.calendar)
        
        # close the new file
        w_nc.close()  
    
    if savecsv:
        df.to_csv(fname+'.csv')
    
    print('Saved files to ' + path)
    return


def ID_exhaust(df,
               co_id = True,
               cn_id = True,
               bc_id = True,
               filter_window = 60*10,
               
               co_stat_window = 10,
               co_num_devs = 4,
#               co_stat_threshold = 0.245,
               
               cn_stat_window = 10,
               cn_num_devs = 4,
#               cn_stat_threshold = 50,
               
               bc_lim = 0.07001
               ):
    print('--- Identifying exhaust...')
    if cn_id:
        print('Using CN to ID exhaust')
#        df = exhaust_flag_rolling_std(df,
        dfcn = exhaust_flag_rolling_var(df.copy(),                                      
                                      column='cn',
                                      stat_window=cn_stat_window,
                                      num_deviations = cn_num_devs
#                                      stat_threshold=cn_stat_threshold,
#                                      filt_around_window=filter_window
                                      )
############################################################################################################################################
###############################################################################
#    cn = df['cn10']
#    ex = df['exhaust']
#    ex.loc[ex.isnull()] = False
#    cn_filt = cn.loc[~ex]
#    plt.plot(cn,'.',cn_filt,'xr',df['cn10_median'],'-k',df['cn10_var_u'],'--k',df['cn10_var_l'],'--k')
#    plt.title('Filter CN')
#    plt.ylim([0,2000])
#    plt.show()    
###############################################################################
############################################################################################################################################
    if co_id:
        print('Using CO to ID exhaust')
#        df = exhaust_flag_rolling_std(df,
        dfco = exhaust_flag_rolling_var(df.copy(),
                                      column='CO',
                                      stat_window=co_stat_window,
                                      num_deviations = co_num_devs
#                                      stat_threshold=co_stat_threshold,
#                                      filt_around_window=filter_window
                                      )
############################################################################################################################################
###############################################################################
#    cn = df['cn10']
#    ex = df['exhaust']
#    ex.loc[ex.isnull()] = False
#    cn_filt = cn.loc[~ex]
#    plt.plot(cn,'.',cn_filt,'xr',df['cn10_median'],'-k',df['cn10_var_u'],'--k',df['cn10_var_l'],'--k')
#    plt.title('Filter CN+CO')
#    plt.ylim([0,2000])
#    plt.show()    
###############################################################################
############################################################################################################################################    
    if bc_id:
        print('Using BC to ID exhaust')
        dfbc = exhaust_flag_bc(df.copy(),
                             bc_lim=bc_lim
#                             filt_around_window=filter_window
                             )
############################################################################################################################################
###############################################################################
#        dfbc_rolling = exhaust_flag_rolling_var(df.copy(),column='BC',stat_window=co_stat_window,num_deviations = co_num_devs)
#    cn = df['cn10']
#    ex = df['exhaust']
#    ex.loc[ex.isnull()] = False
#    cn_filt = cn.loc[~ex]
#    plt.plot(cn,'.',cn_filt,'xr',df['cn10_median'],'-k',df['cn10_var_u'],'--k',df['cn10_var_l'],'--k')
#    plt.title('Filter CN+CO+BC')
#    plt.ylim([0,2000])
#    plt.show()    
#    
#    sum_exh = ex.rolling(window = filter_window,center=True).sum()
#    plt.plot(cn,'.',sum_exh,'o')
#    plt.show()

    ecn = dfcn['exhaust']
    ebc = dfbc['exhaust']
    eco = dfco['exhaust']
    ex = pd.Series([True if any([cn,bc,co]) else False for cn,bc,co in zip(ecn,ebc,eco)],index=eco.index)
    df['exhaust'] = ex
#    cn = df['cn10']
#    cn_f = cn.loc[~ex]
#    cn_ecn = cn.loc[~ecn]
#    cn_eco = cn.loc[~eco]
#    cn_ebc = cn.loc[~ebc]
#    
#    plt.plot(cn,'.',cn_ebc,'.r')
#    plt.title('bc filt only')
#    plt.show()
#    plt.plot(cn,'.',cn_ecn,'.r')
#    plt.title('cn filt only')
#    plt.show()
#    plt.plot(cn,'.',cn_eco,'.r')
#    plt.title('co filt only')
#    plt.show()
#    plt.plot(cn,'.',cn_f,'.r')
#    plt.title('all filters together○')
#    plt.show()
###############################################################################
############################################################################################################################################    
    
    df = filt_surrounding_window(
                                df.copy(),
                                filt_around_window=filter_window
                                )
############################################################################################################################################
###############################################################################
#    cn = df['cn10']
#    ex = df['exhaust']
#    ex.loc[ex.isnull()] = False
#    cn_filt = cn.loc[~ex]
#    sum_exh_c = ex.rolling(window = filter_window,center=True).sum()
#    plt.plot(cn,'.',cn_filt,'xr',df['cn10_median'],'-k',df['cn10_var_u'],'--k',df['cn10_var_l'],'--k',sum_exh_c,'o')
#    plt.title('Filter CN+CO+BC+10min window')
#    plt.ylim([0,2000])
#    plt.show()
#    
#    df.to_hdf('df.h5',key='d')
#    dfbc.to_hdf('dfbc.h5',key='d')
#    dfcn.to_hdf('dfcn.h5',key='d')
#    dfco.to_hdf('dfco.h5',key='d')
#    dfw.to_hdf('dfw.h5',key='d')
    
    ex = df['exhaust']    
    df_filt = df.loc[~ex]
    
    if 'cn10' in df.columns:
        plt.plot(df['cn10'],'.',df_filt['cn10'],'.r')
    else:
        plt.plot(df['cn'],'.',df_filt['cn'],'.r')
    plt.title('CN raw and filt')
    plt.ylim([0,2000])
    plt.show()
    
    plt.plot(df['ccn'],'.',df_filt['ccn'],'.r')
    plt.title('CCN raw and filt')
    plt.ylim([0,2000])
    plt.show()

    plt.plot(df_filt['WindDirRel_vmean'],df_filt['cn10'],'.')
    plt.title('wind dir vs cn10')
    plt.show()
    plt.plot(df_filt['WindDirRel_port'],df_filt['CO'],'.')
    plt.title('wind dir vs co')
    plt.show()
    plt.plot(df_filt['WindDirRel_port'],df_filt['ccn'],'.')
    plt.title('wind dir vs ccn')
    plt.show()
###############################################################################
############################################################################################################################################
    return df
'''
def exhaust_flag_rolling_std(df,
                             column='cn10', 
                             stat_window=10,
                             stat_threshold = 0.03,
                             filt_around_window=1):
    print('--- Identifying exhaust by the standard deviation of '+column)
    df[column+'_std'] = df[column].rolling(window=stat_window,center=True).std()
    #df['cn_stdminute'] = df[column].rolling(window=181,center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df[column+'_std'] > stat_threshold
    # Filter for sec std over threshold
    #exhaust_rows0.loc[(df['cn_stdminute'] > 10.5)] = True
    
    # Filter data around identified periods
    exhaust_rows = exhaust_rows0.rolling(window=filt_around_window,
                                         center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    df.loc[exhaust_rows,'exhaust'] = True
    
    return df
'''

    
def exhaust_flag_bc(df,
                    bc_lim=0.05,
                    filt_around_window=1
                    ):   
    print('--- Identifying exhaust using a BC limit of '+str(bc_lim))
    if 'bc' in df:
        col = 'bc'
    elif 'BC' in df:
        col = 'BC'
    else:
        assert False, 'No BC identified in the data'
    
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    exh = df['exhaust'].copy()
    exh_bc = df[col] > bc_lim
    
    #df.loc[ex_bc,'exhaust'] = True
    df['exhaust'] = [True if (ebc or ex) else False for (ebc, ex) in zip(exh_bc,exh)] #.loc[df[col] > bc_lim] = True
    
    # Filter data around identified periods
#    exhaust_rows = df['exhaust'].rolling(window=filt_around_window,
#                                         center=True).apply(exhaust_window)
#    exhaust_rows.fillna(True,inplace=True)
#    exhaust_rows = exhaust_rows.astype(bool)
#    df.loc[exhaust_rows,'exhaust'] = True
    return df


#@numba.jit
def exhaust_flag_rolling_var(df,
                              column='cn',
                              stat_window=60*10,
                              num_deviations = 4
                              ):
    '''
    Identifies the exhaust using a rolling variability filter. The filter 
    utilises medians and median absolute deviations as these allow more better
    outlier identification since they themselves are insensitive to outliers.
    The function first calculates the rolling MAD and median and then determines
    whether a particular point is outside the range specified by num_deviations
    and if so, identifies it as exhaust. 
    The functional also deals with nan's by filling them in with the population
    median and MAD values, and any values outside 5 MAD's from the median is 
    also replaced by the population values.
    Parameters:
        - df - the dataframe containing the data for use in identification
        - column - the column to use for identification
        - stat_window - the width of the rolling window on which to calculate 
            the median and mad
        - num_deviations - the number of MADs around the median outside which 
            is identified as exhaust. Default 4.
    '''
    if column not in df:
        if column == 'cn':
            if 'cn10' in df:
                column = 'cn10'
            else:
                print('No cn found in the data frame. Skipping cn filter')
                return df
        elif column == 'CO':
            if 'co' in df:
                column = 'co'
            else:
                print('No CO found in the data frame. Skipping CO filter')
                return df
    
        assert column in df, 'Column not found in provided dataframe!'
    
    d = df[column]
    
    # Calculate the rolling MAD
    mad_array = d.rolling(
                          window=stat_window,
                          center=True
                          ).apply(
                                 func=rolling_detrend_stat,
                                 args=('1')
                                 )
    
    # Calculate the global MAD to use when filling in data
    mad0 = mad_array.median()
    mad_glob = get_hist_max(mad_array)
    if mad0 > 5*mad_glob:
        mad = mad_glob
    else:
        mad = mad0
    mad_g_arr = np.empty((len(d),))
    mad_g_arr.fill(mad)
    mad_g_arr = pd.Series(mad_g_arr,index=d.index)
    
    # Calculate the rolling median
    med = d.rolling(window=stat_window,center=True).median()
    
    # Shift the rolling median to the left of the window
    med_leftwindow = np.empty((len(d),))
    med_leftwindow.fill(np.nan)
    med_leftwindow[0:(len(med)-int(stat_window/2))] = med[int(stat_window/2):len(med)]
    med_lw = pd.Series(med_leftwindow,index=med.index)
    # Shift the rolling median to the right of the window
    med_rw = d.rolling(window=stat_window).median()
    
    # Calculate the global median
    med_glob = med.median()

    # Fill in endpoint sections not covered by the window
    med = fill_window_endpoints(med)
    
    # Fill values in polluted areas and nans  
#    med_filled = pd.Series(
#            [med_glob 
#             if (
#                (np.isnan(x)) 
#                or 
#                (x > L + 5*mad)#_g_arr) 
#                or 
#                (x < L - 5*mad)#_g_arr)
##                or
##                (x > R + 5*mad)#_g_arr)
##                or
##                (x < R - 5*mad)#_g_arr)
#                ) 
#             else 
#             x 
#             for x,L,R in zip(med,med_lw,med_rw) ]
#            ,index=med.index)
    # Fill values in polluted areas and nans 
    med_f = med[0]
    med_filled = med.copy()
    for i in range(int(stat_window/2),len(d)-int(stat_window/2),1):
        med_filled[i], med_f = fill_loop(med[i],med_lw[i],med_rw[i],mad,med_f,med_filled)
    for i in range(int(stat_window/2),len(d)-int(stat_window/2),-1):
        med_filled[i], med_f = fill_loop(med[i],med_lw[i],med_rw[i],mad,med_f,med_filled)    
    #med_filled = pd.Series(med_filled,index=med.index)
        
           
#    plt.plot(d,'.k',med+50,'.g',med_rw+5*mad,'--c',med_filled,'.r')  
#    plt.show()
    
    # Create the exhaust boundary definition using MAD
    var_l = med_filled - num_deviations * mad
    var_u = med_filled + num_deviations * mad
    
    # Create the exhaust
    exhaust = pd.Series(
                [(d0 > u) or (d0 < l) for d0,u,l in zip(d,var_u,var_l)]
                ,index=d.index)

    # fill in endpoints created by rolling window    
    exhaust = fill_window_endpoints(exhaust)

    # format correctly
    exhaust = exhaust.astype(bool)
    exhaust = np.array(exhaust)

    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    
    df['exhaust'] = df['exhaust'].astype(bool)
    df.loc[exhaust,'exhaust'] = True
    df[column+'_median'] = med_filled
    df[column+'_var_l'] = var_l
    df[column+'_var_u'] = var_u
    
    return df

@numba.jit
def fill_loop(x,med_lw,med_rw,mad,med_f,med_filled):
    if (
        np.isnan(x)
        or 
        x > med_lw + 5*mad
        or 
        x < med_lw - 5*mad
        or 
        x > med_rw + 5*mad
        or 
        x < med_rw - 5*mad
        ):
        med_filled = med_f
        
    else:
        med_filled = x
        med_f = x
    return med_filled, med_f


def filt_surrounding_window_SLOW(
                            exhaust_df,
                            filt_around_window=60*10
                            ):
    print('Filtering for '+str(filt_around_window/60) + ' minutes around identified exhaust periods')
    ex = exhaust_df['exhaust'].copy()
    ex0 = ex.copy()
    
    truelocs = np.where(ex==True)[0] # choose only the indices where exhaust has been identified.
    for i in truelocs:
        if (i < int(filt_around_window/2)) or (i > len(ex)-int(filt_around_window/2)):
            continue
        
        x = ex0[i-int(filt_around_window/2):i+int(filt_around_window/2)]
        if sum(x) > 0.05*filt_around_window:
            ex.iloc[i-int(filt_around_window/2):i+int(filt_around_window/2)] = True
    '''
    exhaust = exhaust.rolling(
                              window=int(filt_around_window)
                              ,center=True
                              ).apply(
                                      exhaust_window
                                      )
    '''
    exhaust_df['exhaust'] = fill_window_endpoints(ex).astype(bool)
    return exhaust_df



@numba.jit
def filt_surrounding_window(
                            exhaust_df,
                            filt_around_window=60*10
                            ):
    print('Filtering for '+str(filt_around_window/60) + ' minutes around identified exhaust periods')
    ex = exhaust_df['exhaust']
    ex0=ex.copy()
    filt_limit = int(0.05*filt_around_window)
    i0 = int(filt_around_window/2)
    L = len(ex)
    r = range(i0,L-i0)
    
    for i in r:
        x = ex0[i-i0:i+i0]
        if sum(x) > filt_limit:
            ex[i-i0:i+i0] = True
    exhaust_df['exhaust'] = fill_window_endpoints(ex).astype(bool)
    return exhaust_df


def exhaust_window(x):
    '''
    if more than n values within the passed window satisfies the value, then 
    return true.
    
    This is used as a moving window to remove data either side of a filter 
    event
    '''
    if sum(x)>0.05*len(x):
        return True
    else:
        return False   

def get_hist_max(samp):
    n, bins = np.histogram(samp.dropna()[samp<100],500)
    return bins[np.argmax(n)]


def fill_window_endpoints(data):
    if type(data.index[0]) is not int:
        index = data.index
        cols = data.name
        data = data.reset_index()
        data = data.drop('index',axis=1)
        data = data[data.columns[0]]
        reindex = True
    
    
    # Fill in endpoints sections not covered by the window
    i_f = data.first_valid_index()
    i_l = data.last_valid_index()
    data_f = data.iloc[i_f] # first defined median
    data_l = data.iloc[i_l] # last defined median
    for i in range(i_f):
        data.iloc[i] = data_f
    for i in range(i_l,len(data)):
        data.iloc[i] = data_l
    
    if reindex:
        data = np.array(data)
        data = pd.DataFrame(data,index=index,columns=[cols])
        data = data[cols]
        
    return data

def rolling_detrend_stat(x, stat=1):
    '''
    For each window, we detrend it using a least squares regression,
    then calculate the variability on it. 
    stat = 1 gives the median absolute deviation, MAD (default)
    stat = 2 gives the standard deviation
    '''
    x_d = signal.detrend(x)
    if int(stat) == 1:
        return MAD(x_d)
    elif int(stat) == 2:
        return x_d.std()
    else:
        assert False, 'Must choose a stat that is either mad or std'


def MAD(x):
    if type(x) == np.ndarray:
        return 1.4826*np.median(np.abs(x - np.median(x)))
    else:
        return 1.4826*np.abs(x - x.median()).median()
    
    
#==============================================================================
# File IO
#==============================================================================
def load(startdate = '2016-05-06',enddate = '2016-05-07'):
    cn_path, bc_path, co_path = find_data_paths(startdate,enddate)
    #======================================================================
    # Load data
    #======================================================================
    cn = load_cn(cn_path,startdate,enddate)
    bc = load_bc(bc_path,True,startdate,enddate)
    co = load_co(co_path,startdate,enddate)
    
    #======================================================================
    # Merge datasets
    #======================================================================
    df = cn.join(bc,how='outer')
    df = df.join(co,how='outer')
    
    # Get voyage name(s)
    voy = []
    for pth in bc_path:
        v = [x for x in pth.split('\\') if 'in2' in x]
        voy.append(v[0])
        
    return df, voy

def find_data_paths(startdate,enddate, driveletter='r'):
    '''
    Finds the data path for the specified date range. This is required since
    data is divided into voyages.
    '''
    print('Finding data paths to load file...')
    assert startdate is not None, 'You must specify a start and end date!'
    assert enddate is not None, 'You must specify a start and end date!'
    
    # Get a list of voyage folders
    subdirs = [x[0]+'\\' for x in os.walk(driveletter+':\\') if 'underway' in x[0]]
    
    # Extract the voyage name
    #voys = [x.split('/')[2] for x in subdirs]
    
    # Initialise
    new_voys = subdirs
    
    # Check which voyages have already been loaded 
    os.chdir(driveletter+':\\RV_Investigator\\')
    if os.path.isfile('voyage_date_index.csv'):
        vdates = pd.read_csv('voyage_date_index.csv',index_col=0)
        # Compare previous data to current list of voyage folders, then load any new voyage data
        new_voys = [x for x in subdirs if x.split('\\')[2] not in vdates.index]
    
    # Load data from file for each voyage
    for voy_dir in new_voys:
        # Get voyage name
        voy_name = voy_dir.split('\\')[2]
        
        # Find uwy file
        os.chdir(voy_dir)
        fname = glob.glob('*.nc')
        if (len(fname) > 1):
            fname = [x for x in fname if len(x) == len(min(fname,key=len))]
            if len(fname) > 1:
                fname = [x for x in fname if voy_name in x]
                if len(fname) > 1:
                    fname = [fname[0]]
        if (len(fname) == 0):
            print('No nc uwy file in ' + voy_dir)
            continue
        
        # Extract start and end dates from the uwy file
        voy_start, voy_end = read_uwy_dates(fname[0],voy_dir)

        # Append the new voyage dates to the dataframe
        d = pd.DataFrame(
                        {'voyage':voy_name,             
                         'start':voy_start,             
                         'end':voy_end}
                        , index=[0])
        d = d[['voyage','start','end']] # reorder columns
        d = d.set_index('voyage')
        try:
            vdates = vdates.append(d)
        except:
            vdates = d
    if len(new_voys) > 0:
        # Save it to file
        os.chdir(driveletter+':\\RV_Investigator\\')
        vdates = vdates.sort_index()
        vdates.to_csv('voyage_date_index.csv')
            
    # Look up the start and end dates and return the voyage(s) that contain those dates
    sdate = parse(startdate)
    edate = parse(enddate)
    for i in range(0,len(vdates)):
        sdate_i = parse(vdates['start'][i])
        edate_i = parse(vdates['end'][i])
        
        if (sdate > sdate_i) and (sdate < edate_i):
            s_voy_i = i
        if (edate > sdate_i) and (edate < edate_i):
            e_voy_i = i
        try:
            e_voy_i
            break
        except:
            continue
    msg = ' date is not within the available voyage dates. You may have chosen\
a port period where data is unavailable. Please check'
    assert 's_voy_i' in dir(), 'Start' + msg
    assert 'e_voy_i' in dir(), 'End' + msg
    voyages = vdates.index[s_voy_i:e_voy_i+1].tolist()
    
    # Given the voyage(s), you can now return the path for each instrument
    for v in voyages:
        cn_path = [x[0]+'\\' for x in os.walk(driveletter+':\\') if ('cpc' in x[0]) and (v in x[0])]
        bc_path = [x[0]+'\\' for x in os.walk(driveletter+':\\') if ('maap_raw' in x[0]) and (v in x[0])]
        
    co_paths = [x[0]+'\\' for x in os.walk(driveletter+':\\') if ('Aerodyne' in x[0])]
    for i in range(0,len(co_paths)):
        try:
            f_yr = int(co_paths[i].split('\\')[-2])
            if sdate.year == f_yr:
                co_yr_s = i
            if sdate.year == f_yr:
                co_yr_e = i                
        except:
            continue
    assert 'co_yr_s' in dir(), 'Start' + msg
    assert 'co_yr_e' in dir(), 'End' + msg
    co_path = co_paths[co_yr_s:co_yr_e+1]
        
    return cn_path, bc_path, co_path

def read_uwy_dates(fname,voy_dir):
    os.chdir(voy_dir)
    d = nc.Dataset(fname, mode='r')
    # Setup empty data frame using the epoch global attribute from the NetCDF file
    epoch = d.Epoch.split(' ')     
    time0 = epoch[3] + ' ' + epoch[4]
    secDelta = int(float(d.Epoch.split(' seconds since')[0]))  
    sampleSize = d.dimensions['sample'].size
    
    index = pd.date_range(start = time0, freq = '5S', periods = sampleSize) + pd.Timedelta(seconds=secDelta)

    return index[0], index[-1]

def load_cn(cn_path, 
            startdate='2015-01-30',
            enddate='2015-02-02'):
    print('--- Loading CN data...')
    for path in cn_path:
        msg = 'Please ensure you have exported the CPC data to h5 file using \
the CPC processing GUI. Once complete, click OK to continue.\n\
h5 files should be placed in the following folders:\n'
        msg = msg + path + '\n'
        os.chdir(path)
        if len(glob.glob('*.h5'))==0:
            Mbox('CN data',msg,0)
    dfc0=[]
    for path in cn_path:
        os.chdir(path)
        secfiles = glob.glob('*.h5')
        secfiles = [x for x in secfiles if len(x)==len(min(secfiles,key=len))]
        dfc = []
        for file in secfiles:
            cn_temp = pd.read_hdf(file, key='cn')
            dfc.append(cn_temp)
        # Concatenate each df in the dictionary
        dfc = pd.concat(dfc)
        
        dfc0.append(dfc)
    df = pd.concat(dfc0)
    df = df.sort_index()
    df = df[startdate:enddate].copy()
    df.rename(columns={'Concentration':'cn'},inplace=True)
    print('CN data loaded!')
    return df



def Mbox(title, text, style=1):
    ''' Show messagebox '''
    ctypes.windll.user32.MessageBoxW(0, text, title, style)


def load_bc(bc_path, load_from_uwy = True,
            startdate='2015-01-30',
            enddate='2015-02-02'):
    ''' Load bc from file, dealing with any new additions when added'''
    print('--- Loading BC data...')
    
    if load_from_uwy:
        df = load_bc_from_uwy(bc_path,startdate,enddate)
        print('BC data loaded!')
        return df
    else:
        df = load_bc_from_file(bc_path,startdate,enddate)
        if len(df) == 0:
            for path in bc_path:
                os.chdir(path)
                files = glob.glob('*.h5')
                for f in files:
                    os.remove(f)
            df = load_bc_from_file(bc_path,startdate,enddate)
        return df

def load_bc_from_file(bc_path, 
            startdate='2015-01-30',
            enddate='2015-02-02'):
    dfbc0=[]
    for path in bc_path:
        os.chdir(path)
        bc_h5 = glob.glob('*.h5')
        if len(bc_h5)==0:
            load_bc_to_h5(bc_path)
            bc_h5 = glob.glob('*.h5')
        df = []
        for file in bc_h5:
            bc_temp = pd.read_hdf(file, key='bc')
            df.append(bc_temp)
        # Concatenate each df in the dictionary
        df = pd.concat(df)
        dfbc0.append(df)
    dfbc = pd.concat(dfbc0)
    dfbc.sort_index()
    dfbc.index.tz = None
    dfbc.columns = ['bc']
    dfbc = dfbc[startdate:enddate].copy()
    print('BC data loaded!')
    return dfbc

def load_bc_to_h5(bc_path, 
            startdate='2015-01-30',
            enddate='2015-02-02'):

    import netCDF4 as nc
    
    for path in bc_path:
        os.chdir(path)
        filelist = glob.glob('*.absphoto')    
        filelist.sort()
        for f in filelist:
            d = nc.Dataset(f, mode='r')
            
            # Setup empty data frame using the epoch global attribute from the NetCDF file
            sampleSize = d.dimensions['time'].size
            bc0 = pd.DataFrame(index = np.arange(0,sampleSize))
            
            # Extract time information
            
            bc0 = bc0.assign(daysSince = d.variables['time'][:]) 
            date0 = parse(['time',d.variables['time'].units][1].split('days since ')[1])
            bc0['Timestamp'] = [date0 + pd.Timedelta(days=x) for x in bc0['daysSince']]
            
            bc0 = bc0.assign(bc_ngm3 = d.variables['black_carbon_conc'][:]) 
            try:
                bc = bc.append(bc0)
            except:
                bc = bc0
            d.close()
            print('Loaded ' + f + ' from file')
      
    bc = bc.set_index('Timestamp')
    del bc['daysSince']
    bc = bc.sort_index()
    bc.to_hdf('bc_raw_' + bc_path[0].split('\\')[2]+'.h5',key='bc')
    return

def load_bc_from_uwy(bc_path, 
            startdate='2015-01-30',
            enddate='2015-02-02'):
    
    print('Loading BC from UWY data')
    
    os.chdir(bc_path[-1]+'..')
    flist = glob.glob('*bc*.h5')
    if len(flist)>0:
        df = pd.read_hdf(flist[0],key='bc')
        print('BC data loaded from h5 file')
    else:
        df = []
        for path in bc_path:
            # Get the underway folder
#            fldlist = path.split('\\')
#            path = fldlist[0] + '\\'
#            for f in range(1, fldlist.index('underway')+1):
#                path = path+fldlist[f]+'\\'
            
            os.chdir(path+'..')
            
            nc_flist = glob.glob('*uwy*.nc')
            nc_file = min(nc_flist,key=len)
            
            d = nc.Dataset(nc_file, mode='r')
            
            # Setup empty data frame using the epoch global attribute from the NetCDF file
            epoch = d.Epoch.split(' ')     
            time0 = epoch[3] + ' ' + epoch[4]
            secDelta = int(float(d.Epoch.split(' seconds since')[0]))  
            sampleSize = d.dimensions['sample'].size
            
            index = pd.date_range(start = time0, freq = '5S', periods = sampleSize) + pd.Timedelta(seconds=secDelta)
            dfu = pd.DataFrame(index = index)
            
            dfu = dfu.assign(bc = d.variables['blackCarbonConc'][:])  
            
            df.append(dfu)
        df = pd.concat(df)
        df.sort_index()
        
        df = df.resample('1S').interpolate()
            
        df.to_hdf('bc_from_uwy.h5',key='bc')
        print('Saved bc_from_uwy.h5 to '+ path)
    df = df[startdate:enddate].copy()
    
    return df



def load_co(
            local_path = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\', 
            startdate='2015-01-30',
            enddate='2015-02-02'
            ):
    ''' Loads CO data from raw files '''
    if type(local_path) == str:
        local_path = [local_path]
    print('--- Loading CO data...')
    df = []
    for path in local_path:
        df_temp = ghg.read_aerodyne_data(
                    local_path_aer=path , 
                    startdate = startdate, 
                    enddate = enddate, 
                    abridged = True)
        if df_temp is not None:
            df.append(df_temp)
        
    df = pd.concat(df)
    df.sort_index()
    df = ghg.resample_interpolation(df)
    print('CO data loaded!')
    return df

# if this script is run at the command line, run the main script   
if __name__ == '__main__': 
	main()