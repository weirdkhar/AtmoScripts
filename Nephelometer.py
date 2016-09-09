# -*- coding: utf-8 -*-
"""
This script contains function that:
- load RV Investigator Underway data from it's output netCDF file
or if this has already been done, from its working HDF5 file.

- Creates masks from neph parameters for use in removing exhaust polluted data

@author: hum094
"""
def Load(directoryPath, From_HDF_or_NC, filt_or_raw = 'filt', NC_filename = 'None'):
    import sys
    sys.path.append('c:\\Dropbox\\RuhiFiles\\Research\\ProgramFiles\\pythonfiles\\')

    ###
    import netCDF4 as nc
    
    import pandas as pd
    import os
    import glob
    
    os.chdir(directoryPath)
    
    if From_HDF_or_NC == 'HDF':
        if filt_or_raw == 'filt':
            h5_filename = 'neph_filt.h5'
        elif filt_or_raw == 'raw':
            h5_filename = 'neph_raw.h5'
        else:
            print("Error: Don't know which h5 file to load! Please ensure your input for filt_or_raw is either filt or raw.")
            return
        neph = pd.read_hdf(h5_filename, key='neph')
        neph_units = pd.read_hdf('neph_units.h5',key='neph_units')
    
    elif From_HDF_or_NC == 'NC':
        
        filelist = glob.glob('*.neph')
        
        for i in range(len(filelist)):
            NC_filename = filelist[i]
            d = nc.Dataset(NC_filename, mode='r')
            
            # Setup the timestamp index from the NetCDF file
            day0 = pd.to_datetime(d['time'].units.split('days since ')[1])
            day0python = pd.to_datetime(0)
            daysSince = d.variables['time'][:]
            
            timestamp = pd.to_datetime(daysSince, unit='d') - day0python + day0
            
            columns = ['BC_mask','wind_mask','misc_mask']
            neph = pd.DataFrame(index = timestamp, columns = columns)
            neph = neph.fillna(1) # Fill mask with 1s rather than NaNs
            
            del columns
            
            neph = neph.assign(baroPress = d.variables['baroPress'][:])                            
            neph = neph.assign(enclTemp = d.variables['enclTemp'][:])                           
            neph = neph.assign(relHumidity = d.variables['relHumidity'][:])       			 
            neph = neph.assign(sampleTemp = d.variables['sampleTemp'][:])				 
            
            neph = neph.assign(w450nm0deg = d.variables['w450nm0deg'][:])
            neph = neph.assign(w450nm10deg = d.variables['w450nm10deg'][:]) 
            neph = neph.assign(w450nm15deg = d.variables['w450nm15deg'][:]) 
            neph = neph.assign(w450nm20deg = d.variables['w450nm20deg'][:]) 
            neph = neph.assign(w450nm25deg = d.variables['w450nm25deg'][:]) 
            neph = neph.assign(w450nm30deg = d.variables['w450nm30deg'][:]) 
            neph = neph.assign(w450nm35deg = d.variables['w450nm35deg'][:]) 
            neph = neph.assign(w450nm40deg = d.variables['w450nm40deg'][:]) 
            neph = neph.assign(w450nm45deg = d.variables['w450nm45deg'][:]) 
            neph = neph.assign(w450nm50deg = d.variables['w450nm50deg'][:]) 
            neph = neph.assign(w450nm55deg = d.variables['w450nm55deg'][:]) 
            neph = neph.assign(w450nm60deg = d.variables['w450nm60deg'][:]) 
            neph = neph.assign(w450nm65deg = d.variables['w450nm65deg'][:]) 
            neph = neph.assign(w450nm70deg = d.variables['w450nm70deg'][:]) 
            neph = neph.assign(w450nm75deg = d.variables['w450nm75deg'][:]) 
            neph = neph.assign(w450nm80deg = d.variables['w450nm80deg'][:]) 
            neph = neph.assign(w450nm85deg = d.variables['w450nm85deg'][:]) 
            neph = neph.assign(w450nm90deg = d.variables['w450nm90deg'][:])
    		
            neph = neph.assign(w525nm0deg = d.variables['w525nm0deg'][:])
            neph = neph.assign(w525nm10deg = d.variables['w525nm10deg'][:]) 
            neph = neph.assign(w525nm15deg = d.variables['w525nm15deg'][:]) 
            neph = neph.assign(w525nm20deg = d.variables['w525nm20deg'][:]) 
            neph = neph.assign(w525nm25deg = d.variables['w525nm25deg'][:]) 
            neph = neph.assign(w525nm30deg = d.variables['w525nm30deg'][:]) 
            neph = neph.assign(w525nm35deg = d.variables['w525nm35deg'][:]) 
            neph = neph.assign(w525nm40deg = d.variables['w525nm40deg'][:]) 
            neph = neph.assign(w525nm45deg = d.variables['w525nm45deg'][:]) 
            neph = neph.assign(w525nm50deg = d.variables['w525nm50deg'][:]) 
            neph = neph.assign(w525nm55deg = d.variables['w525nm55deg'][:]) 
            neph = neph.assign(w525nm60deg = d.variables['w525nm60deg'][:]) 
            neph = neph.assign(w525nm65deg = d.variables['w525nm65deg'][:]) 
            neph = neph.assign(w525nm70deg = d.variables['w525nm70deg'][:]) 
            neph = neph.assign(w525nm75deg = d.variables['w525nm75deg'][:]) 
            neph = neph.assign(w525nm80deg = d.variables['w525nm80deg'][:]) 
            neph = neph.assign(w525nm85deg = d.variables['w525nm85deg'][:]) 
            neph = neph.assign(w525nm90deg = d.variables['w525nm90deg'][:])
    		
            neph = neph.assign(w635nm0deg = d.variables['w635nm0deg'][:])
            neph = neph.assign(w635nm10deg = d.variables['w635nm10deg'][:]) 
            neph = neph.assign(w635nm15deg = d.variables['w635nm15deg'][:]) 
            neph = neph.assign(w635nm20deg = d.variables['w635nm20deg'][:]) 
            neph = neph.assign(w635nm25deg = d.variables['w635nm25deg'][:]) 
            neph = neph.assign(w635nm30deg = d.variables['w635nm30deg'][:]) 
            neph = neph.assign(w635nm35deg = d.variables['w635nm35deg'][:]) 
            neph = neph.assign(w635nm40deg = d.variables['w635nm40deg'][:]) 
            neph = neph.assign(w635nm45deg = d.variables['w635nm45deg'][:]) 
            neph = neph.assign(w635nm50deg = d.variables['w635nm50deg'][:]) 
            neph = neph.assign(w635nm55deg = d.variables['w635nm55deg'][:]) 
            neph = neph.assign(w635nm60deg = d.variables['w635nm60deg'][:]) 
            neph = neph.assign(w635nm65deg = d.variables['w635nm65deg'][:]) 
            neph = neph.assign(w635nm70deg = d.variables['w635nm70deg'][:]) 
            neph = neph.assign(w635nm75deg = d.variables['w635nm75deg'][:]) 
            neph = neph.assign(w635nm80deg = d.variables['w635nm80deg'][:]) 
            neph = neph.assign(w635nm85deg = d.variables['w635nm85deg'][:]) 
            neph = neph.assign(w635nm90deg = d.variables['w635nm90deg'][:]) 
            
            try:
                data
            except NameError:
                data = neph.copy() # data doesn't exist yet, initialise
            else:
                # data has been initialised, therefore, just append.
                data = data.append(neph)
            
            d.close()
        
        
        # Units
        NC_filename = filelist[0]
        d = nc.Dataset(NC_filename, mode='r')
        
        neph_units = pd.DataFrame(columns=['parameter','units'])
        neph_units.loc[1] = ['baroPress', d.variables['baroPress'].units]                            
        neph_units.loc[2] = ['enclTemp', d.variables['enclTemp'].units]                           
        neph_units.loc[3] = ['relHumidity', d.variables['relHumidity'].units]       			 
        neph_units.loc[4] = ['sampleTemp', d.variables['sampleTemp'].units]				 
        
        neph_units.loc[5] = ['w450nm0deg', d.variables['w450nm0deg'].units]
        neph_units.loc[6] = ['w450nm10deg', d.variables['w450nm10deg'].units] 
        neph_units.loc[7] = ['w450nm15deg', d.variables['w450nm15deg'].units] 
        neph_units.loc[8] = ['w450nm20deg', d.variables['w450nm20deg'].units] 
        neph_units.loc[9] = ['w450nm25deg', d.variables['w450nm25deg'].units] 
        neph_units.loc[10] = ['w450nm30deg', d.variables['w450nm30deg'].units] 
        neph_units.loc[11] = ['w450nm35deg', d.variables['w450nm35deg'].units] 
        neph_units.loc[12] = ['w450nm40deg', d.variables['w450nm40deg'].units] 
        neph_units.loc[13] = ['w450nm45deg', d.variables['w450nm45deg'].units] 
        neph_units.loc[14] = ['w450nm50deg', d.variables['w450nm50deg'].units] 
        neph_units.loc[15] = ['w450nm55deg', d.variables['w450nm55deg'].units] 
        neph_units.loc[16] = ['w450nm60deg', d.variables['w450nm60deg'].units] 
        neph_units.loc[17] = ['w450nm65deg', d.variables['w450nm65deg'].units] 
        neph_units.loc[18] = ['w450nm70deg', d.variables['w450nm70deg'].units] 
        neph_units.loc[19] = ['w450nm75deg', d.variables['w450nm75deg'].units] 
        neph_units.loc[20] = ['w450nm80deg', d.variables['w450nm80deg'].units] 
        neph_units.loc[21] = ['w450nm85deg', d.variables['w450nm85deg'].units] 
        neph_units.loc[22] = ['w450nm90deg', d.variables['w450nm90deg'].units]
		
        neph_units.loc[23] = ['w525nm0deg', d.variables['w525nm0deg'].units]
        neph_units.loc[24] = ['w525nm10deg', d.variables['w525nm10deg'].units] 
        neph_units.loc[25] = ['w525nm15deg', d.variables['w525nm15deg'].units] 
        neph_units.loc[26] = ['w525nm20deg', d.variables['w525nm20deg'].units] 
        neph_units.loc[27] = ['w525nm25deg', d.variables['w525nm25deg'].units] 
        neph_units.loc[28] = ['w525nm30deg', d.variables['w525nm30deg'].units] 
        neph_units.loc[29] = ['w525nm35deg', d.variables['w525nm35deg'].units] 
        neph_units.loc[30] = ['w525nm40deg', d.variables['w525nm40deg'].units] 
        neph_units.loc[31] = ['w525nm45deg', d.variables['w525nm45deg'].units] 
        neph_units.loc[32] = ['w525nm50deg', d.variables['w525nm50deg'].units] 
        neph_units.loc[33] = ['w525nm55deg', d.variables['w525nm55deg'].units] 
        neph_units.loc[34] = ['w525nm60deg', d.variables['w525nm60deg'].units] 
        neph_units.loc[35] = ['w525nm65deg', d.variables['w525nm65deg'].units] 
        neph_units.loc[36] = ['w525nm70deg', d.variables['w525nm70deg'].units] 
        neph_units.loc[37] = ['w525nm75deg', d.variables['w525nm75deg'].units] 
        neph_units.loc[38] = ['w525nm80deg', d.variables['w525nm80deg'].units] 
        neph_units.loc[39] = ['w525nm85deg', d.variables['w525nm85deg'].units] 
        neph_units.loc[40] = ['w525nm90deg', d.variables['w525nm90deg'].units]
		
        neph_units.loc[41] = ['w635nm0deg', d.variables['w635nm0deg'].units]
        neph_units.loc[42] = ['w635nm10deg', d.variables['w635nm10deg'].units] 
        neph_units.loc[43] = ['w635nm15deg', d.variables['w635nm15deg'].units] 
        neph_units.loc[44] = ['w635nm20deg', d.variables['w635nm20deg'].units] 
        neph_units.loc[45] = ['w635nm25deg', d.variables['w635nm25deg'].units] 
        neph_units.loc[46] = ['w635nm30deg', d.variables['w635nm30deg'].units] 
        neph_units.loc[47] = ['w635nm35deg', d.variables['w635nm35deg'].units] 
        neph_units.loc[48] = ['w635nm40deg', d.variables['w635nm40deg'].units] 
        neph_units.loc[49] = ['w635nm45deg', d.variables['w635nm45deg'].units] 
        neph_units.loc[50] = ['w635nm50deg', d.variables['w635nm50deg'].units] 
        neph_units.loc[51] = ['w635nm55deg', d.variables['w635nm55deg'].units] 
        neph_units.loc[52] = ['w635nm60deg', d.variables['w635nm60deg'].units] 
        neph_units.loc[53] = ['w635nm65deg', d.variables['w635nm65deg'].units] 
        neph_units.loc[54] = ['w635nm70deg', d.variables['w635nm70deg'].units] 
        neph_units.loc[55] = ['w635nm75deg', d.variables['w635nm75deg'].units] 
        neph_units.loc[56] = ['w635nm80deg', d.variables['w635nm80deg'].units] 
        neph_units.loc[57] = ['w635nm85deg', d.variables['w635nm85deg'].units] 
        neph_units.loc[58] = ['w635nm90deg', d.variables['w635nm90deg'].units] 
        
        d.close()
        
        # Save to HDF        
        data.to_hdf('neph_raw.h5',key='neph')
        neph_units.to_hdf('neph_units.h5',key='neph_units')
        
        
    else:
        print("Error: Don't know what file to load! Please ensure your input for From_HDF_or_NC is either HDF or NC.")
        return
    return (neph, neph_units)
    
neph_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/Neph/'
Load(neph_path,'NC','raw')