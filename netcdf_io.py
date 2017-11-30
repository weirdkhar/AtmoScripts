'''

General functions for the load and saving of netcdf files

'''
import netCDF4 as nc
import pandas as pd

def read_nc(nc_filename, 
            outputDir = None, output_filename = None, output_h5_key = None,
            time_col = 'time', time_unit_parser = 'days since '
            ):
    d_nc = nc.Dataset(nc_filename, mode='r')
    
    # Setup the timestamp index from the NetCDF file
    day0 = pd.to_datetime(d_nc['time'].units.split('days since ')[1])
    day0python = pd.to_datetime(0)
    daysSince = d_nc.variables['time'][:]
    
    timestamp = pd.to_datetime(daysSince, unit='d') - day0python + day0
    
    columns = list(d_nc.variables.keys())
    d = pd.DataFrame(index = timestamp, columns = columns)
    d = d.fillna(1) # Fill mask with 1s rather than NaNs
    
    for c in columns:
        d[c] = d_nc.variables[c[:]]
     
    
    if outputDir is not None:
        if output_filename is None:
            output_fname = outputDir + '/' + nc_filename.split('/')[-1].split('.')[0]+'.h5'
        else:
            output_fname= outputDir + '/' + output_filename +'.h5'
        
        d.to_hdf(output_fname, key = output_h5_key)
    
    d_nc.close()
    return d