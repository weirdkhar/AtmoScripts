# -*- coding: utf-8 -*-
"""
This script contains function that:
- load RV Investigator Underway data from it's output netCDF file
or if this has already been done, from its working HDF5 file.

- Creates masks from UWY parameters for use in removing exhaust polluted data

@author: hum094
"""
import os
import pandas as pd
import glob

def Load(directoryPath, From_HDF_or_NC, filt_or_raw = 'filt', NC_filename = 'None'):
    import sys
    sys.path.append('c:\\Dropbox\\RuhiFiles\\Research\\ProgramFiles\\pythonfiles\\')

    ###
    import netCDF4 as nc
    
    import pandas as pd
    import os
    
    os.chdir(directoryPath)
    
    if From_HDF_or_NC.upper() == 'HDF':
        if filt_or_raw == 'filt':
            h5_filename = 'uwy_filt.h5'
        elif filt_or_raw == 'raw':
            h5_filename = 'uwy_raw.h5'
        else:
            print("Error: Don't know which h5 file to load! Please ensure your input for filt_or_raw is either filt or raw.")
            return
        uwy = pd.read_hdf(h5_filename, key='uwy')
        uwy_units = pd.read_hdf('uwy_units.h5',key='uwy_units')
    
    elif From_HDF_or_NC.upper() == 'NC':
        
        d = nc.Dataset(NC_filename, mode='r')
        
        # Setup empty data frame using the epoch global attribute from the NetCDF file
        epoch = d.Epoch.split(' ')     
        time0 = epoch[3] + ' ' + epoch[4]
        secDelta = int(float(d.Epoch.split(' seconds since')[0]))  
        sampleSize = d.dimensions['sample'].size
        
        index = pd.date_range(start = time0, freq = '5S', periods = sampleSize) + pd.Timedelta(seconds=secDelta)
        uwy = pd.DataFrame(index = index)
        
        
        # Met data
        uwy = uwy.assign(lat = d.variables['latitude'][:])                            
        uwy = uwy.assign(lon = d.variables['longitude'][:])                           
        
        uwy = uwy.assign(atmPress = d.variables['atmPressure'][:])       			 
        
        uwy = uwy.assign(WindGust_max = d.variables['maxWindGust'][:])				 
        uwy = uwy.assign(WindDirRel_port = d.variables['portRelWindDir'][:])          
        uwy = uwy.assign(WindDirRel_stbd = d.variables['stbdRelWindDir'][:])          
        uwy = uwy.assign(WindSpdRel_port = d.variables['portRelWindSpeed'][:])        
        uwy = uwy.assign(WindSpdRel_stbd = d.variables['stbdRelWindSpeed'][:])        
        uwy = uwy.assign(WindDirTru_port = d.variables['portTrueWindDir'][:])         
        uwy = uwy.assign(WindDirTru_stbd = d.variables['stbdTrueWindDir'][:])         
        uwy = uwy.assign(WindSpdTru_port = d.variables['portTrueWindSpeed'][:])       
        uwy = uwy.assign(WindSpdTru_stbd = d.variables['stbdTrueWindSpeed'][:])       
        
        uwy = uwy.assign(WindDirRel_Ultrasonic = d.variables['ultraRelWindDir'][:])   
        uwy = uwy.assign(WindSpdRel_Ultrasonic = d.variables['ultraRelWindSpeed'][:]) 
        uwy = uwy.assign(WindDirTru_Ultrasonic = d.variables['ultraTrueWindDir'][:])  
        uwy = uwy.assign(WindSpdTru_Ultrasonic = d.variables['ultraTrueWindSpeed'][:])
        
        uwy = uwy.assign(AirTemp_port = d.variables['portAirTemp'][:])                
        uwy = uwy.assign(AirTemp_stbd = d.variables['stbdAirTemp'][:])                
        uwy = uwy.assign(RH_port = d.variables['portHumidity'][:])                    
        uwy = uwy.assign(RH_stbd = d.variables['stbdHumidity'][:])                    
        
        uwy = uwy.assign(PAR_port = d.variables['portPAR'][:])                        
        uwy = uwy.assign(PAR_stbd = d.variables['stbdPAR'][:])                        
        uwy = uwy.assign(Pyra_port = d.variables['portPyranometer'][:])               
        uwy = uwy.assign(Pyra_stbd = d.variables['stbdPyranometer'][:])               
        uwy = uwy.assign(Radi_port = d.variables['portRadiometer'][:])                
        uwy = uwy.assign(Radi_stbd = d.variables['stbdRadiometer'][:])                
        
        uwy = uwy.assign(rainAccum = d.variables['rain'][:])                          
        
        # Ship data                                                  
        uwy = uwy.assign(depth = d.variables['depth'][:])                             
                                                                     
        uwy = uwy.assign(COG = d.variables['courseOG'][:])                            
        uwy = uwy.assign(shipHeading = d.variables['shipHeading'][:])                 
        uwy = uwy.assign(gyroHeading = d.variables['gyroHeading'][:])                 
        uwy = uwy.assign(lonGndSpd = d.variables['longitudinalGroundSpeed'][:])       
        uwy = uwy.assign(lonWtrSpd = d.variables['longitudinalWaterSpeed'][:])        
        uwy = uwy.assign(spdOG = d.variables['speedOG'][:])                           
        uwy = uwy.assign(transverseGndSpd = d.variables['transverseGroundSpeed'][:])  
        uwy = uwy.assign(transverseWtrSpd = d.variables['transverseWaterSpeed'][:])   
                                                                   
        uwy = uwy.assign(inletBearing = d.variables['inletBearing'][:])               
        
        # Aerosol data                                               
        uwy = uwy.assign(BC = d.variables['blackCarbonConc'][:])                      
        
        # Reactive gas data                                          
        uwy = uwy.assign(O3_1 = d.variables['o3Ozone1'][:])                           
        uwy = uwy.assign(O3_2 = d.variables['o3Ozone2'][:])                           
        
        # GHG data                                                   
        uwy = uwy.assign(ch4 = d.variables['ch4Dry'][:])                              
        uwy = uwy.assign(co2 = d.variables['co2Dry'][:])                              
        uwy = uwy.assign(h2o = d.variables['h2O'][:])                                 
        
        # Ocean data                                                 
        uwy = uwy.assign(flu = d.variables['fluorescence'][:])                        
        uwy = uwy.assign(SeaTemp = d.variables['isarWaterTemp'][:])                   
        uwy = uwy.assign(salinity = d.variables['salinity'][:])   
        
        
        # Units
        uwy_units = pd.DataFrame(columns=['parameter','units'])
        uwy_units.loc[1] = ['latitude',d.variables['latitude'].units]
        uwy_units.loc[2] = ['longitude',d.variables['longitude'].units]
        
        uwy_units.loc[3] = ['atmPressure',d.variables['atmPressure'].units]
        uwy_units.loc[4] = ['maxWindGust',d.variables['maxWindGust'].units]
        uwy_units.loc[5] = ['portRelWindDir',d.variables['portRelWindDir'].units]
        uwy_units.loc[6] = ['stbdRelWindDir',d.variables['stbdRelWindDir'].units]
        uwy_units.loc[7] = ['portRelWindSpeed',d.variables['portRelWindSpeed'].units]
        uwy_units.loc[8] = ['stbdRelWindSpeed',d.variables['stbdRelWindSpeed'].units]
        uwy_units.loc[9] = ['portTrueWindDir',d.variables['portTrueWindDir'].units]
        uwy_units.loc[10] = ['stbdTrueWindDir',d.variables['stbdTrueWindDir'].units]
        uwy_units.loc[11] = ['portTrueWindSpeed',d.variables['portTrueWindSpeed'].units]
        uwy_units.loc[12] = ['stbdTrueWindSpeed',d.variables['stbdTrueWindSpeed'].units]
        uwy_units.loc[13] = ['ultraRelWindDir',d.variables['ultraRelWindDir'].units]
        uwy_units.loc[14] = ['ultraRelWindSpeed',d.variables['ultraRelWindSpeed'].units]
        uwy_units.loc[15] = ['ultraTrueWindDir',d.variables['ultraTrueWindDir'].units]
        uwy_units.loc[16] = ['ultraTrueWindSpeed',d.variables['ultraTrueWindSpeed'].units]
        
        uwy_units.loc[17] = ['portAirTemp',d.variables['portAirTemp'].units]
        uwy_units.loc[18] = ['stbdAirTemp',d.variables['stbdAirTemp'].units]
        uwy_units.loc[19] = ['portHumidity',d.variables['portHumidity'].units]
        uwy_units.loc[20] = ['stbdHumidity',d.variables['stbdHumidity'].units]
        
        uwy_units.loc[21] = ['portPAR',d.variables['portPAR'].units]
        uwy_units.loc[22] = ['stbdPAR',d.variables['stbdPAR'].units]
        uwy_units.loc[23] = ['portPyranometer',d.variables['portPyranometer'].units]
        uwy_units.loc[24] = ['stbdPyranometer',d.variables['stbdPyranometer'].units]
        uwy_units.loc[25] = ['portRadiometer',d.variables['portRadiometer'].units]
        uwy_units.loc[26] = ['stbdRadiometer',d.variables['stbdRadiometer'].units]
        
        uwy_units.loc[27] = ['rain',d.variables['rain'].units]
        
        uwy_units.loc[28] = ['depth',d.variables['depth'].units]
        
        uwy_units.loc[29] = ['courseOG',d.variables['courseOG'].units]
        uwy_units.loc[30] = ['shipHeading',d.variables['shipHeading'].units]
        uwy_units.loc[31] = ['gyroHeading',d.variables['gyroHeading'].units]
        uwy_units.loc[32] = ['longitudinalGroundSpeed',d.variables['longitudinalGroundSpeed'].units]
        uwy_units.loc[33] = ['longitudinalWaterSpeed',d.variables['longitudinalWaterSpeed'].units]
        uwy_units.loc[34] = ['speedOG',d.variables['speedOG'].units]
        uwy_units.loc[35] = ['transverseGroundSpeed',d.variables['transverseGroundSpeed'].units]
        uwy_units.loc[36] = ['transverseWaterSpeed',d.variables['transverseWaterSpeed'].units]
        
        uwy_units.loc[37] = ['inletBearing',d.variables['inletBearing'].units]
        
        uwy_units.loc[38] = ['blackCarbonConc',d.variables['blackCarbonConc'].units]
        
        uwy_units.loc[39] = ['o3Ozone1',d.variables['o3Ozone1'].units]
        uwy_units.loc[40] = ['o3Ozone2',d.variables['o3Ozone2'].units]
        
        uwy_units.loc[41] = ['ch4Dry',d.variables['ch4Dry'].units]
        uwy_units.loc[42] = ['co2Dry',d.variables['co2Dry'].units]
        uwy_units.loc[43] = ['h2O',d.variables['h2O'].units]
        
        uwy_units.loc[44] = ['fluorescence',d.variables['fluorescence'].units]
        uwy_units.loc[45] = ['isarWaterTemp',d.variables['isarWaterTemp'].units]
        uwy_units.loc[46] = ['salinity',d.variables['salinity'].units]
        
        
        d.close()
        
        uwy.to_hdf('uwy_raw.h5',key='uwy')
        uwy_units.to_hdf('uwy_units.h5',key='uwy_units')
        
    else:
        print("Error: Don't know what file to load! Please ensure your input for From_HDF_or_NC is either HDF or NC.")
        return
    return (uwy, uwy_units)
  
def parse_date_index(data):
    # Change index to datetime
    try:
        index = pd.to_datetime(data['Date/Time'], format = '%Y-%m-%d %H:%M')
    except ValueError:
        index = pd.to_datetime(data['Date/Time'], format = '%d/%m/%Y %H:%M')
    data = data.set_index(index)
    
    return data
  
def Load_CSV_UWY_to_HDF(uwy_path,earlyVoyageNames=False):
    
    
    os.chdir(uwy_path)

    filelist = glob.glob('*.csv')
    data = pd.read_csv(filelist[0]) # Initialise
    data = parse_date_index(data)
    datafile = pd.HDFStore('uwy_temp.h5')
    
    for i in range(1,len(filelist)):
        try:        
            data_temp = pd.read_csv(filelist[i])
            data_temp = parse_date_index(data_temp)
            data = data.append(data_temp)

#            current_fname = 'uwy_raw_temp_'+str(i)+'of'+str(len(filelist))#+'.h5'
#            data.to_hdf(current_fname, key='uwy')
#            
#            # Delete previous file        
#            if os.path.isfile('uwy_raw_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5'):
#                os.remove('uwy_raw_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5')
            
            current_key = 'uwy_raw_temp_'+str(i)+'of'+str(len(filelist))
            previous_key = 'uwy_raw_temp_'+str(i-1)+'of'+str(len(filelist))                       
            datafile.put(current_key,data, data_columns=True)
            try: 
                datafile.remove(previous_key)
            except KeyError:
                continue
            
            
        
        except ValueError:
            if os.path.isfile('uwy_raw_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5'):
                os.remove('uwy_raw_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5')
            continue
    datafile.close()
    
    # Rename columns to align with the loading from the NetCDF file
    if earlyVoyageNames:
        data.rename(columns={'Air Sampling Inlet Bearing (degree)' : 'inletBearing',
                        'Air Temperature (degC)' : 'AirTemp',
                        'Air flow rate (L/h)' : 'AirInletFlowRate',
                        'Atmospheric Pressure (mbar)' : 'atmPress',
                        'CH4 Concentration (ppm)' : 'ch4',
                        'CH4 dry Concentration (ppm)' : 'ch4_dry',
                        'CO2 Concentration (ppm)' : 'co2',
                        'CO2 dry Concentration (ppm)' : 'co2_dry',
                        'Concentration of black carbon (ug/m^3)' : 'BC',
                        'Corrected Wind Direction (degree)' : 'WindDirTru',
                        'Corrected Wind Speed (knot)' : 'WindSpdTru',
                        'Course Over Ground (degree)' : 'COG',
                        'Fluorescence (dimensionless)' : 'flu',
                        'Latitude (degree_north)' : 'lat',
                        'Longitude (degree_east)' : 'lon',
                        'Longitudinal Ground Speed (m/s)' : 'lonGndSpd',
                        'Longitudinal Water Speed (m/s)' : 'lonWtrSpd',
                        'Maximum Wind Gust (knot)' : 'WindGust_max',
                        'Ozone (ppb)' : 'O3_1',
                        'Ozone (ppb).1' : 'O3_2',
                        'Ozone1 Meter flags' : 'O3_1_flags',
                        'Ozone2 Meter flags' : 'O3_2_flags',
                        'Port Air Temperature (degC)' : 'AirTemp_port',
                        'Port Humidity (%)' : 'RH_port',
                        'Salinity (PSU)' : 'salinity',
                        'Ship Heading (degree)' : 'shipHeading',
                        'Speed Over Ground (knot)' : 'spdOG',
                        'Starboard Air Temperature (degC)' : 'AirTemp_stbd',
                        'Starboard Humidity (%)' : 'RH_stbd',
                        'Transverse Ground Speed (m/s)' : 'transverseGndSpd',
                        'Transverse Water Speed (m/s)' : 'transverseWtrSpd',
                        'Uncorrected Wind Direction (degree)' : 'WindDirRel',
                        'Uncorrected Wind Speed (knot)' : 'WindSpdRel',
                        'Water Temperature (degC)' : 'SeaTemp',
                        'cumulative hour rainfall (mm)' : 'rainAccum',
                        'current rainfall rate for the foremast optical rain gauge (mm/h)' : 'rainRate',
                        'depth (m)' : 'depth',
                        'port incoming long wave radiation (W/m^2)' : 'Radi_port',
                        'port incoming photosynthetically active radiation (uE/m^2/s)' : 'PAR_port',
                        'port incoming short wave radiation (W/m^2)' : 'Pyra_port',
                        'starboard incoming long wave radiation (W/m^2)' : 'Radi_stbd',
                        'starboard incoming photosynthetically active radiation (uE/m^2/s)' : 'PAR_stbd',
                        'starboard incoming short wave radiation (W/m^2)' : 'Pyra_stbd',
                        'water temperature (degree_Celsius)' : 'seaTemp_2'                        
                        })
        # drop columns not in use
        if 'CO2 Pump speed (l/min)' in data:
            data.drop('CO2 Pump speed (l/min)', axis=1,inplace=True)
        if 'Condenser temperature (degC)' in data:
            data.drop('Condenser temperature (degC)', axis=1,inplace=True)
        if 'Drop Keel system fault' in data:
            data.drop('Drop Keel system fault', axis=1,inplace=True)
        if 'Equilibrator pressure (hPa)' in data:
            data.drop('Equilibrator pressure (hPa)', axis=1,inplace=True)
        if 'Equilibrator water temperature (degC)' in data:
            data.drop('Equilibrator water temperature (degC)', axis=1,inplace=True)
        if 'Fluorescence (dimensionless)' in data:
            data.drop('Fluorescence (dimensionless)', axis=1,inplace=True)
        if 'Licor flow (ml/min)' in data:
            data.drop('Licor flow (ml/min)', axis=1,inplace=True)
        if 'Licor pressure (hPa)' in data:
            data.drop('Licor pressure (hPa)', axis=1,inplace=True)
        if 'Lock on Ground' in data:
            data.drop('Lock on Ground', axis=1,inplace=True)
        if 'Lock on water' in data:
            data.drop('Lock on water', axis=1,inplace=True)
        if 'Port keel is locked' in data:
            data.drop('Port keel is locked', axis=1,inplace=True)
        if 'Sensor Temperature (degC)' in data:
            data.drop('Sensor Temperature (degC)', axis=1,inplace=True)
        if 'Starboard keel is locked' in data:
            data.drop('Starboard keel is locked', axis=1,inplace=True)
        if 'TSG Flow Rate (l/min)' in data:
            data.drop('TSG Flow Rate (l/min)', axis=1,inplace=True)
        if 'Tracking Target Bearing (degree)' in data:
            data.drop('Tracking Target Bearing (degree)', axis=1,inplace=True)
        if 'Vent flow (ml/min)' in data:
            data.drop('Vent flow (ml/min)', axis=1,inplace=True)
        if 'Water Concentration Percentage (dimensionless)' in data:
            data.drop('Water Concentration Percentage (dimensionless)', axis=1,inplace=True)
        if 'Water flow (l/min)' in data:
            data.drop('Water flow (l/min)', axis=1,inplace=True)
        if 'Water vapour (mmol/mole)' in data:
            data.drop('Water vapour (mmol/mole)', axis=1,inplace=True)
        if 'XCO2 (ppm)' in data:
            data.drop('XCO2 (ppm)', axis=1,inplace=True)
        if 'anemometer air heading (degree)' in data:
            data.drop('anemometer air heading (degree)', axis=1,inplace=True)
        if 'anemometer air speed (knot)' in data:
            data.drop('anemometer air speed (knot)', axis=1,inplace=True)
        if 'conductivity (mS/cm)' in data:
            data.drop('conductivity (mS/cm)', axis=1,inplace=True)
        if 'foremast optical cumulative hour rainfall (mm)' in data:
            data.drop('foremast optical cumulative hour rainfall (mm)', axis=1,inplace=True)
        if 'port drop keel extension (m)' in data:
            data.drop('port drop keel extension (m)', axis=1,inplace=True)
        if 'starboard drop keel extension (m)' in data:
            data.drop('starboard drop keel extension (m)', axis=1,inplace=True)
        if 'water flow in branch (l/min)' in data:
            data.drop('water flow in branch (l/min)', axis=1,inplace=True)
        if 'water flow in main (l/min)' in data:
            data.drop('water flow in main (l/min)', axis=1,inplace=True)
        if 'water pressure (bar)' in data:
            data.drop('water pressure (bar)', axis=1,inplace=True)
        if 'water temperature (degree_Celsius)' in data:
            data.drop('water temperature (degree_Celsius)' , axis=1,inplace=True)
            
    else:
        data.rename(columns={'Port Air Temperature (degC)' : 'AirTemp_port',
                        'Starboard Air Temperature (degC)' : 'AirTemp_stbd',
                        'Atmospheric Pressure (mbar)' : 'atmPress',
                        'Course Over Ground (degree)' : 'COG',
                        'Depth (m)' : 'depth',
                        'Fluorescence (dimensionless)' : 'flu',
                        'Latitude (degree_north)' : 'lat',
                        'Longitude (degree_east)' : 'lon',
                        'Longitudinal Ground Speed (m/s)' : 'lonGndSpd',
                        'Longitudinal Water Speed (m/s)' : 'lonWtrSpd',
                        'Port PAR (uE/m^2/s)' : 'PAR_port',
                        'Starboard PAR (uE/m^2/s)' : 'PAR_stbd',
                        'Port Pyranometer (W/m^2)' : 'Pyra_port',
                        'Starboard Pyranometer (W/m^2)' : 'Pyra_stbd',
                        'Port Radiometer (W/m^2)' : 'Radi_port',
                        'Starboard Radiometer (W/m^2)' : 'Radi_stbd',
                        'Accumulated Hourly Rain (mm)' : 'rainAccum',
                        'Port Humidity (%)' : 'RH_port',
                        'Starboard Humidity (%)' : 'RH_stbd',
                        'Salinity (PSU)' : 'salinity',
                        'Water Temperature (degC)' : 'SeaTemp',
                        'Ship Heading (degree)' : 'shipHeading',
                        'Speed Over Ground (knot)' : 'spdOG',
                        'Transverse Ground Speed (m/s)' : 'transverseGndSpd',
                        'Transverse Water Speed (m/s)' : 'transverseWtrSpd',
                        'Port Relative Wind Direction (degree)' : 'WindDirRel_port',
                        'Starboard Relative Wind Direction (degree)' : 'WindDirRel_stbd',
                        'Ultrasonic Relative Wind Direction (degree)' : 'WindDirRel_Ultrasonic',
                        'Port True Wind Direction (degree)' : 'WindDirTru_port',
                        'Starboard True Wind Direction (degree)' : 'WindDirTru_stbd',
                        'Ultrasonic True Wind Direction (degree)' : 'WindDirTru_Ultrasonic',
                        'Maximum Wind Gust (knot)' : 'WindGust_max',
                        'Port Relative Wind Speed (knot)' : 'WindSpdRel_port',
                        'Starboard Relative Wind Speed (knot)' : 'WindSpdRel_stbd',
                        'Ultrasonic Relative Wind Speed (knot)' : 'WindSpdRel_Ultrasonic',
                        'Port True Wind Speed (knot)' : 'WindSpdTru_port',
                        'Starboard True Wind Speed (knot)' : 'WindSpdTru_stbd',
                        'Ultrasonic True Wind Speed (knot)' : 'WindSpdTru_Ultrasonic',
                        'Air Flow Rate (L/h)' : 'AirInletFlowRate',
                        'Mode of the Air Sampling Inlet' : 'AirInletMode',
                        'Air Sampling Inlet Bearing (degree)' : 'inletBearing',
                        'Concentration of Black Carbon (ug/m^3)' : 'BC',
                        'Absorption Photometer Status' : 'BC_status',
                        'CH4 Dry Concentration (ppm)' : 'ch4',
                        'CO2 Dry Concentration (ppm)' : 'co2',
                        'Ozone1 (ppb)' : 'O3_1',
                        'Ozone1 Meter flags' : 'O3_1_flags',
                        'Ozone2 (ppb)' : 'O3_2',
                        'Ozone2 Meter flags' : 'O3_2_flags'
                        },inplace=True)
    
        data_units_dict = {'AirTemp_port' : 'degC',
                    'AirTemp_stbd' : 'degC',
                    'atmPress' : 'mbar',
                    'COG' : 'degree',
                    'depth' : 'm',
                    'flu' : 'dimensionless',
                    'lat' : 'degree_north',
                    'lon' : 'degree_east',
                    'lonGndSpd' : 'm/s',
                    'lonWtrSpd' : 'm/s',
                    'PAR_port' : 'uE/m^2/s',
                    'PAR_stbd' : 'uE/m^2/s',
                    'Pyra_port' : 'W/m^2',
                    'Pyra_stbd' : 'W/m^2',
                    'Radi_port' : 'W/m^2',
                    'Radi_stbd' : 'W/m^2',
                    'rainAccum' : 'mm',
                    'RH_port' : '%',
                    'RH_stbd' : '%',
                    'salinity' : 'PSU',
                    'SeaTemp' : 'degC',
                    'shipHeading' : 'degree',
                    'spdOG' : 'knot',
                    'transverseGndSpd' : 'm/s',
                    'transverseWtrSpd' : 'm/s',
                    'WindDirRel_port' : 'degree',
                    'WindDirRel_stbd' : 'degree',
                    'WindDirRel_Ultrasonic' : 'degree',
                    'WindDirTru_port' : 'degree',
                    'WindDirTru_stbd' : 'degree',
                    'WindDirTru_Ultrasonic' : 'degree',
                    'WindGust_max' : 'knot',
                    'WindSpdRel_port' : 'knot',
                    'WindSpdRel_stbd' : 'knot',
                    'WindSpdRel_Ultrasonic' : 'knot',
                    'WindSpdTru_port' : 'knot',
                    'WindSpdTru_stbd' : 'knot',
                    'WindSpdTru_Ultrasonic' : 'knot',
                    'AirInletFlowRate' : 'L/h',
                    'AirInletMode' : '',
                    'BC' : 'ug/m^3',
                    'BC_status' : '',
                    'ch4' : 'ppm',
                    'co2' : 'ppm',
                    'inletBearing' : 'degree',
                    'O3_1' : 'ppb',
                    'O3_1_flags' : '',
                    'O3_2' : 'ppb',
                    'O3_2_flags' : ''
                    }
    #Convert to dataframe
    data_units_series = pd.Series(data_units_dict, name='units')
    data_units_series.index.name='parameter'
    data_units = data_units_series.reset_index()
    
    
    # Drop columns not in use
    if 'Altitude (m)' in data:
        data.drop('Altitude (m)', axis=1, inplace=True)
    if 'Lock On Water' in data:
        data.drop('Lock On Water', axis=1, inplace=True)
    if 'Lock on Ground' in data:
        data.drop('Lock on Ground', axis=1, inplace=True)
    if 'TSG Sensor Temperature (degC)' in data:
        data.drop('TSG Sensor Temperature (degC)', axis=1, inplace=True)
    if 'TSG Flow Rate (l/min)' in data:
        data.drop('TSG Flow Rate (l/min)', axis=1, inplace=True)
    if 'UWY Lab Main Flow (l/min)' in data:
        data.drop('UWY Lab Main Flow (l/min)', axis=1, inplace=True)
    if 'UWY Lab Branch Flow (l/min)' in data:
        data.drop('UWY Lab Branch Flow (l/min)', axis=1, inplace=True)
    if 'Gyro Heading (degree)' in data:
        data.drop('Gyro Heading (degree)', axis=1, inplace=True)
    if 'Equilibrator Water Temperature (degC)' in data:
        data.drop('Equilibrator Water Temperature (degC)', axis=1, inplace=True)
    if 'XCO2 (ppm)' in data:
        data.drop('XCO2 (ppm)', axis=1, inplace=True)
    if 'Port Drop Keel Extension (m)' in data:
        data.drop('Port Drop Keel Extension (m)', axis=1, inplace=True)
    if 'Starboard Drop Keel Extension (m)' in data:
        data.drop('Starboard Drop Keel Extension (m)', axis=1, inplace=True)
    if 'Tracking Target Bearing (degree)' in data:
        data.drop('Tracking Target Bearing (degree)', axis=1, inplace=True)
    if 'Transverse Ground Speed (knot)' in data:
        data.drop('Transverse Ground Speed (knot)', axis=1, inplace=True)
    if 'Transverse Water Speed (knot)' in data:
        data.drop('Transverse Water Speed (knot)', axis=1, inplace=True)
    if 'Water Concentration Percentage (dimensionless)' in data:
        data.drop('Water Concentration Percentage (dimensionless)', axis=1, inplace=True)
    if 'Water Vapour (mmol/mole)' in data:
        data.drop('Water Vapour (mmol/mole)', axis=1, inplace=True)
    if 'CO2 Pump Speed (l/min)' in data:
        data.drop('CO2 Pump Speed (l/min)', axis=1, inplace=True)
    if 'Condenser Temperature (degC)' in data:
        data.drop('Condenser Temperature (degC)', axis=1, inplace=True)
    if 'Equilibrator  Pressure (hPa)' in data:
        data.drop('Equilibrator  Pressure (hPa)', axis=1, inplace=True)
    if 'ISAR Water Temperature (degC)' in data:
        data.drop('ISAR Water Temperature (degC)', axis=1, inplace=True)
    if 'Licor Pressure (hPa)' in data:
        data.drop('Licor Pressure (hPa)', axis=1, inplace=True)
    if 'Licor flow (ml/min)' in data:
        data.drop('Licor flow (ml/min)', axis=1, inplace=True)
    if 'Longitudinal Ground Speed (knot)' in data:
        data.drop('Longitudinal Ground Speed (knot)', axis=1, inplace=True)
    if 'Longitudinal Water Speed (knot)' in data:
        data.drop('Longitudinal Water Speed (knot)', axis=1, inplace=True)
    if 'Vent Flow (ml/min)' in data:
        data.drop('Vent Flow (ml/min)', axis=1, inplace=True)
    if 'Water Flow (l/min)' in data:
        data.drop('Water Flow (l/min)', axis=1, inplace=True)
    if 'heave (m)' in data:
        data.drop('heave (m)', axis=1, inplace=True)
    if 'pitch (degree)' in data:
        data.drop('pitch (degree)', axis=1, inplace=True)
    if 'roll (degree)' in data:
        data.drop('roll (degree)', axis=1, inplace=True)
    
                            
    # Delete final file and save normally
    data.to_hdf('uwy_raw.h5',key='uwy')
    data_units.to_hdf('uwy_units.h5',key='uwy_units')
    os.remove('uwy_temp.h5')
        
    return
    
def rename_columns_MaidenVoyage(uwy_data):
    
    uwy_data.rename(columns={
                            'Air Temperature (degC)': 'AirTemp',
                            'Air flow rate (L/h)' : 'AirInletFlowRate',
                            'CH4 Concentration (ppm)' : 'ch4',
                            'CH4 dry Concentration (ppm)': 'ch4_dry',
                            'CO2 Concentration (ppm)' : 'co2', 
                            'CO2 dry Concentration (ppm)' : 'co2_dry', 
                            'Concentration of black carbon (ug/m^3)' : 'BC_uwy',
                            'Corrected Wind Direction (degree)' : 'WindDirTru',
                            'Corrected Wind Speed (knot)' : 'WindSpdTru', 
                            'Date/Time' : 'Date/Time_uwy',
                            'Ozone (ppb)' : 'O3_1', 
                            'Ozone (ppb).1' : 'O3_2',
                            'Uncorrected Wind Direction (degree)': 'WindDirRel', 
                            'Uncorrected Wind Speed (knot)' : 'WindSpdRel',
                            'anemometer air heading (degree)' : 'WindDir_anem',
                            'anemometer air speed (knot)' : 'WindSpd_anem',
                            'conductivity (mS/cm)' : 'conductivity',
                            'cumulative hour rainfall (mm)' : 'rainAccum',
                            'current rainfall rate for the foremast optical rain gauge (mm/h)' : 'rainRate',
                            'depth (m)' : 'depth', 
                            'foremast optical cumulative hour rainfall (mm)' : 'rainAccum_optical',
                            'port incoming long wave radiation (W/m^2)' : 'Radi_port',
                            'port incoming photosynthetically active radiation (uE/m^2/s)' : 'PAR_port',
                            'port incoming short wave radiation (W/m^2)' : 'Pyra_port',
                            'starboard incoming long wave radiation (W/m^2)' : 'Radi_stbd',
                            'starboard incoming photosynthetically active radiation (uE/m^2/s)' : 'PAR_stbd',
                            'starboard incoming short wave radiation (W/m^2)' : 'Pyra_stbd',
                            'water temperature (degree_Celsius)' : 'seaTemp'                     
                        },inplace=True)
    uwy_data.drop([
                  'CO2 Pump speed (l/min)', 
                  'Condenser temperature (degC)', 
                  'Drop Keel system fault',
                  'Equilibrator  pressure (hPa)', 
                  'Equilibrator water temperature (degC)',
                  'Licor pressure (hPa)', 
                  'Lock on water',
                  'Port keel is locked',
                  'Sensor Temperature (degC)', 
                  'Starboard keel is locked',
                  'Vent flow (ml/min)', 
                  'Water flow (l/min)',
                  'Water vapour (mmol/mole)', 
                  'port drop keel extension (m)',
                  'starboard drop keel extension (m)', 
                  'water flow in branch (l/min)',
                  'water flow in main (l/min)',
                  'water pressure (bar)'
                  ], axis=1, inplace=True)   
                  
    return uwy_data   
#def create_uwy_masks(DataPath,
#                     apply_mask_to_create_filt_dataset,
#                     BC_lim = 0.05, 
#                     wind_dir_mask = False, windDir_lLim = 110, windDir_uLim = 260,
#                     radon_mask = False, 
#                     wind_sensor_disagreement_mask = False,
#                     SaveToFile = True):
#    """
#    Creates masks based on:
#        - Black carbon (BC) data, which is a proxy for ship
#        exhaust.
#        - wind direction
#        - Radon
#        - Disagreement between port and starboard wind sensors
#    """
#
#    import pandas as pd
#    import numpy as np
#    
#    (uwy,uwy_units) = Load(DataPath,'HDF','raw')
#    
#    
#    #Create BC mask
#    uwy.loc[uwy.BC > BC_lim, 'BC_mask'] = np.nan
#    uwy.loc[uwy.BC < -0.7, 'BC_mask'] = np.nan
#    
#    # Check out how well the filtering worked
#    #uwy = uwy.assign(BC_filt = uwy.BC * uwy.BC_mask)
#    #uwy = uwy.assign(O3_filt = uwy.O3_1 * uwy.BC_mask)
#    
#    #uwy.plot(x='BC',y='O3_1',kind='scatter')
#    #uwy.plot(x='BC_filt',y='O3_1',kind='scatter')
#    
#    #uwy['BC_filt'].hist(bins=100)
#    
#    #uwy['O3_1'].hist(bins=100)
#    #uwy['O3_filt'].hist(bins=100)
#     
#    # Wind Direction Mask
#    if wind_dir_mask:
#        uwy.loc[(uwy.WindDirRel_port < windDir_uLim) & (uwy.WindDirRel_port > windDir_lLim), 'WindDir_mask'] = np.nan
#        uwy.loc[(uwy.WindDirRel_stbd < windDir_uLim) & (uwy.WindDirRel_stbd > windDir_lLim), 'WindDir_mask'] = np.nan
#        
#        
#    # Wind sensor disagreement mask
#    if wind_sensor_disagreement_mask:
#        uwy = uwy.assign(WindDirRatio1 = uwy.WindDirRel_port/uwy.WindDirRel_stbd)
#        uwy = uwy.assign(WindDirRatio2 = uwy.WindDirRel_stbd/uwy.WindDirRel_port)
#        uwy.loc[uwy.WindDirRatio1 > 1.5, 'wind_sensor_mask'] = np.nan
#        uwy.loc[uwy.WindDirRatio2 > 1.5, 'wind_sensor_mask'] = np.nan
#    
#    #uwy = uwy.assign(WindDirRel_port_filt = uwy.WindDirRel_port * uwy.wind_mask)
#    #uwy = uwy.assign(WindDirRel_stbd_filt = uwy.WindDirRel_stbd * uwy.wind_mask)
#    
#    #uwy.plot(x='WindDirRel_port',y='WindDirRel_stbd',kind='scatter',xlim=[-10,370],ylim=[-10,370])
#    #uwy.plot(x='WindDirRel_port_filt',y='WindDirRel_stbd_filt',kind='scatter',xlim=[-10,370],ylim=[-10,370])
#    
#    
#    # filter    
#    
#    
#    #Create filtered dataset
#    if apply_mask_to_create_filt_dataset:
##        mask = uwy[['BC_mask']]
##        if wind_dir_mask:
##            WD_mask = uwy[['WindDir_mask']]
##            mask = pd.DataFrame(mask.values * WD_mask.values,
##                                columns = mask.columns, index = uwy.index)
##        if wind_sensor_disagreement_mask:
##            WndDisagree_mask = uwy[['wind_sensor_mask']]
##            mask = pd.DataFrame(mask.values * WndDisagree_mask.values,
##                                columns = mask.columns, index = uwy.index)
#        mask = uwy['BC_mask']
#        if wind_dir_mask:
#            WD_mask = uwy['WindDir_mask']
#            mask = mask * WD_mask
#        if wind_sensor_disagreement_mask:
#            WndDisagree_mask = uwy['wind_sensor_mask']
#            mask = mask * WndDisagree_mask
#        
#        Create_UWY_Filt(DataPath, mask)
#    
#    if SaveToFile:
#        #Save data to file
#        uwy.to_hdf('uwy_raw.h5',key='uwy')
#        return None
#    else:
#        return uwy



#def Create_UWY_Filt(DataPath,mask):
#    import pandas as pd
#    
#    mask = pd.Series.to_frame(mask) #Convert series to frame 
#    # Load raw data from H5 file
#    (uwy,uwy_units) = Load(DataPath,'HDF','raw')
#    
#    #Initialise new dataset
#    uwy_filt = uwy[['BC_mask']].copy() #Initialise
#    uwy_filt.columns = ['mask'] #rename column
#    
#    # get column names for new dataset
#    names = uwy.columns.values
#    
#    for i in range(1,len(uwy.columns)): # for each column
#        #Extract column to apply mask to        
#        col = uwy[[names[i]]] 
#        # Apply mask to column using element-wise multiplication
#        df_temp = pd.DataFrame(mask.values * col.values, columns = col.columns, index = col.index)
#        # Append filtered column to new dataset
#        uwy_filt = uwy_filt.join(df_temp)
#    
#    #Save new dataset
#    uwy_filt.to_hdf('uwy_filt.h5',key='uwy')
#    
#    return
    
    

    
def Create_WndDir_VectorMean(uwy_data):
    import numpy as np  
    import os
    import pandas as pd
    
    # Check if the variable is already there in the file, if not, create it    
    if 'WindDirRel_vmean' not in uwy_data:
        
        # Check if the filtered dataset already exists, if it does, import it, otherwise, create it
        if os.path.isfile('uwy_filt.h5'):
            uwy_filt = pd.read_hdf('uwy_filt.h5',key='uwy')
            uwy_data.insert(uwy_data.columns.get_loc('WindDirRel_stbd')+1,'WindDirRel_vmean',uwy_filt['WindDirRel_vmean'])
            uwy_data.insert(uwy_data.columns.get_loc('WindDirTru_stbd')+1,'WindDirTru_vmean',uwy_filt['WindDirTru_vmean'])
        
        else:
    
            # Vector addition taking into account the step change at 360 and 0
            variable = ['Rel','Tru']
            for j in range(2):
                var = variable[j]
                # Insert vector to the column to the right of WindDirRel_stbd
                uwy_data.insert(uwy_data.columns.get_loc('WindDir'+var+'_stbd')+1,'WindDir'+var+'_vmean',np.nan)
        
                for i in range(len(uwy_data)):
                    # wind speed
                    WS_p = uwy_data['WindSpd'+var+'_port'][i]
                    WS_s = uwy_data['WindSpd'+var+'_stbd'][i] 
            
                    # wind direction
                    WD_p = uwy_data['WindDir'+var+'_port'][i]
                    WD_s = uwy_data['WindDir'+var+'_stbd'][i]
                    
                    if abs(WD_p - WD_s) > 270:
                        if WD_p < WD_s:
                            WD_p = WD_p + 360
                        else:
                            WD_s = WD_s + 360
            
                    # Create wind direction vector weighted by wind speed
                    WD_vect = WD_s*WS_s/(WS_s+WS_p) + WD_p*WS_p/(WS_s+WS_p)
                    
                    # remove 360 addition where it was added.
                    if WD_vect > 360:
                        WD_vect = WD_vect - 360
                    
                    # Assign vector to dataframe column        
                    uwy_data.ix[i,'WindDir'+var+'_vmean'] = WD_vect            
            
    
    return uwy_data
    

	

def create_exhaust_mask(uwy_data, 
                          Filter4WindDir = True, Filter4BC = True, Filter4CNstd = True, Filter4O3 = True, FiltSpan = 0,
                          mask_level_num = 1, 
                          WD_exhaust_upper_limit = 277, WD_exhaust_lower_limit = 115,
                          BC_lim = 0.05,
                          O3_min = 12,
                          CN_std_ID = 'cn3 and cn10',
                          CN3_std_lim = 120, CN10_std_lim = 120,
                          manual_exhaust_mask = []
                          ):
    '''
    Create empty column for exhaust marker then populate
       # Level 1 is a heavy removal of all problematic data - this uses a wide wind direction filter and BC filter.
       # Level 2 serves to save good data that Level 1 has removed    
    '''
    import numpy as np
    
    
    if mask_level_num == 1:
        mask = 'exhaust_mask_L1'
        if mask not in uwy_data:
            # Insert new column of 1s at the beginning, ensuring the order is correct of L1, L2, then L3
            uwy_data.insert(0,mask,np.ones(len(uwy_data),dtype=np.int))
    elif mask_level_num == 2:
        mask = 'exhaust_mask_L2'
        if mask not in uwy_data:        
            # Insert new column of 1s at the beginning
            if 'exhaust_mask_L1' in uwy_data:
                uwy_data.insert(1,mask,np.ones(len(uwy_data),dtype=np.int))
            else:
                uwy_data.insert(0,mask,np.ones(len(uwy_data),dtype=np.int))
    elif mask_level_num == 3:
        mask = 'exhaust_mask_L3'
        if mask not in uwy_data:
            # Insert new column of 1s at the beginning
            if 'exhaust_mask_L1' in uwy_data:
                if 'exhaust_mask_L2' in uwy_data:
                    uwy_data.insert(2,mask,np.ones(len(uwy_data),dtype=np.int))
                else:
                    uwy_data.insert(1,mask,np.ones(len(uwy_data),dtype=np.int))            
            else:
                uwy_data.insert(0,mask,np.ones(len(uwy_data),dtype=np.int))
    else:
        print('Unsure which processing level we are modifying')
        return
    
    
     
    if Filter4BC:
        uwy_data.loc[uwy_data['BC'] > BC_lim, mask] = np.nan
    
    if Filter4WindDir:
        if 'WindDirRel_vmean' in uwy_data:
            uwy_data.loc[(uwy_data['WindDirRel_vmean'] > WD_exhaust_lower_limit) & (uwy_data['WindDirRel_vmean'] < WD_exhaust_upper_limit), mask] = np.nan
        elif 'WindDirRel' in uwy_data:
            uwy_data.loc[(uwy_data['WindDirRel'] > WD_exhaust_lower_limit) & (uwy_data['WindDirRel'] < WD_exhaust_upper_limit), mask] = np.nan
        else:
            print('Dont know what the relative wind direction parameter is! Please check!')

    if Filter4O3:
        uwy_data.loc[uwy_data['O3_2'] < O3_min, mask] = np.nan
    
    # Filter for standard deviation of CN
    if Filter4CNstd:
        if CN_std_ID.lower() == 'cn3 and cn10':
            cn3_filt = True
            cn10_filt = True
        elif CN_std_ID.lower() == 'cn3':
            cn3_filt = True
            cn10_filt = False
        elif CN_std_ID.lower() == 'cn10':
            cn3_filt = False
            cn10_filt = True
        else:
            print("Can't interpret the CN filter to create the exhaust mask! Please check input")
            return
        
        if cn3_filt:
            uwy_data.loc[uwy_data['CN3_std'] > CN3_std_lim, mask] = np.nan
        if cn10_filt:
            uwy_data.loc[uwy_data['CN10_std'] > CN10_std_lim, mask] = np.nan
    
    # Check if any manual filter entries exist. If so, add them to the mask
    if len(manual_exhaust_mask) > 0:
        # work through mask periods and set values to nan
        for i in range(int(len(manual_exhaust_mask)/2)):
            uwy_data.loc[(uwy_data.index >= manual_exhaust_mask[2*i]) & (uwy_data.index < manual_exhaust_mask[2*i+1])]= np.nan
        
    # Filter for a period of time around each exhaust period
    if FiltSpan > 0:
        span = 12*FiltSpan
        i = 0
        while i < len(uwy_data)-span:
            # Set values to nan the minute before a period is identified
            if np.isnan(uwy_data[mask].iloc[i]):
                if np.isfinite(uwy_data[mask].iloc[i+1]): # end of period
                    uwy_data[mask].iloc[i:i+span] = np.nan
                    i = i+span
    
                elif np.isfinite(uwy_data[mask].iloc[i-1]):
                    uwy_data[mask].iloc[i-span:i] = np.nan # start of period
                    i = i+1
    
                else:
                    i = i+1
            else:
                i =i+1
        
            
    return uwy_data


def UWY_QC(uwy_data, mask_period_timestamp_list = None):
    import numpy as np
    try:
        # Keep Wind direction data within range.
        uwy_data.loc[(uwy_data.WindDirRel_port < 0) | (uwy_data.WindDirRel_port > 360), 'WindDirRel_port'] = np.nan
    except:
        pass
    try:    
        uwy_data.loc[(uwy_data.WindDirRel_stbd < 0) | (uwy_data.WindDirRel_stbd > 360), 'WindDirRel_stbd'] = np.nan
    except:
        pass
    try:    
        uwy_data.loc[(uwy_data.WindDirRel < 0) | (uwy_data.WindDirRel > 360), 'WindDirRel_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindDirTru_port < 0) | (uwy_data.WindDirTru_port > 360), 'WindDirTru_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindDirTru_stbd < 0) | (uwy_data.WindDirTru_stbd > 360), 'WindDirTru_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindDirTru < 0) | (uwy_data.WindDirTru > 360), 'WindDirTru_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindDirRel_Ultrasonic < 0) | (uwy_data.WindDirRel_Ultrasonic  > 360), 'WindDirRel_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindDirTru_Ultrasonic < 0) | (uwy_data.WindDirTru_Ultrasonic  > 360), 'WindDirTru_Ultrasonic'] = np.nan
    except:
        pass
    
        
        # Keep Wind speed data positive
    try:
        uwy_data.loc[(uwy_data.WindSpdRel_port < 0), 'WindSpdRel_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdRel_stbd < 0), 'WindSpdRel_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdRel < 0), 'WindSpdRel_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdTru_port < 0), 'WindSpdTru_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdTru_stbd < 0), 'WindSpdTru_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdTru < 0), 'WindSpdTru_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdRel_Ultrasonic < 0), 'WindSpdRel_Ultrasonic'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindSpdTru_Ultrasonic < 0), 'WindSpdTru_Ultrasonic'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.WindGust_max < 0), 'WindGust_max'] = np.nan
    except:
        pass
    
        
        # Atmospheric pressure postive
    try:
        uwy_data.loc[(uwy_data.atmPress < 0), 'atmPress'] = np.nan
    except:
        pass
    
        
        # Lat long within bounds
    try:
        uwy_data.loc[(uwy_data.lat < -90) | (uwy_data.lat > 90), 'lat'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.lon < -180) | (uwy_data.lon > 180), 'lon'] = np.nan
    except:
        pass
        
        # Temp
    try:
        uwy_data.loc[(uwy_data.AirTemp_port < -273.15), 'AirTemp_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.AirTemp_stbd < -273.15), 'AirTemp_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.SeaTemp < -273.15), 'SeaTemp'] = np.nan    
    except:
        pass
        
        # RH
    try:
        uwy_data.loc[(uwy_data.RH_port < 0) | (uwy_data.RH_port > 100), 'RH_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.RH_stbd < 0) | (uwy_data.RH_port > 100), 'RH_stbd'] = np.nan
    except:
        pass
        
        # Radiation
    try:    
        uwy_data.loc[(uwy_data.PAR_port < 0), 'PAR_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.Pyra_port < 0), 'Pyra_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.Radi_port < 0), 'Radi_port'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.PAR_stbd < 0), 'PAR_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.Pyra_stbd < 0), 'Pyra_stbd'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.Radi_stbd < 0), 'Radi_stbd'] = np.nan
    except:
        pass

        
        # Rain
    try:
        uwy_data.loc[(uwy_data.rainAccum < 0), 'rainAccum'] = np.nan
    except:
        pass
        
        # Ocean parameters
    try:
       uwy_data.loc[(uwy_data.salinity < 30), 'salinity'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.flu < -1), 'flu'] = np.nan
    except:
        pass
        
        #Ship stuff
    try:
        uwy_data.loc[(uwy_data.depth < 0), 'depth'] = np.nan
    except:
        pass
    try:
        uwy_data.loc[(uwy_data.COG < 0) | (uwy_data.COG > 360), 'COG'] = np.nan
    except:
        pass
    
    #    transverseGndSpd
    #    transverseWtrSpd
    try:
        uwy_data.loc[(uwy_data.inletBearing < 0) | (uwy_data.inletBearing > 360), 'inletBearing'] = np.nan
    except:
        pass 
    
    
    if mask_period_timestamp_list is not None:
        # work through mask periods and set values to nan
        for i in range(int(len(mask_period_timestamp_list)/2)):
            uwy_data.loc[(uwy_data.index >= mask_period_timestamp_list[2*i]) & (uwy_data.index < mask_period_timestamp_list[2*i+1])]= np.nan
            
    return uwy_data
    
def resample_timebase(data=0,RawDataPath='',input_h5_filename='',variable='uwy',time_int='default'):
    ### Time resampling

    import pandas as pd
    import os
    
    input_h5_filename = input_h5_filename.split('.')[0]

    if not isinstance(data, pd.DataFrame): #if no data provided, try to load from file
        if (not RawDataPath == '') & (not input_h5_filename == ''):
            os.chdir(RawDataPath)
            if os.path.isfile(input_h5_filename+'.h5'): 
                data = pd.read_hdf(input_h5_filename+'.h5', key=variable)
            else:
                print("Your input filename does not exist! Please check")
                return
        else:
            print("Please input either a dataframe or a datapath and filename where data can be found")
            return
    
    
    # define time resampling intervals unless specified in function input
    if time_int == 'default':
        time_int = ['5S','1Min', '5Min', '10Min', '30Min', '1H', '3H', '6H', '12H', '1D']    
    if type(time_int) is str:
        time_int = [time_int]
    
    # define time resampling intervals
    i_lim = len(time_int)
    for i in range(0, i_lim):
        t_int = time_int[i]    
        # Initialise    
        data_resamp = data.resample(t_int,fill_method=None).mean()
        data_resamp['count'] = data['lat'].resample(t_int,fill_method=None, how=['count'])
        
        # Save to file
        if isinstance(data,pd.DataFrame):        
            outputfilename = variable+'_'+time_int[i].lower()+'.h5'
        else:
            outputfilename = input_h5_filename+'_'+ time_int[i].lower() +'.h5'
        data_resamp.to_hdf(outputfilename, key=variable)
    
    return data_resamp

def UWY_LoadandProcess(uwy_path,
                       uwy_filename=None,
                       filtOrRaw='raw', 
                       timeResolution='',
                       mask_period_timestamp_list = None,
                       earlyVoyageNames = False
                       ):
    import pandas as pd
    import glob    
    import os
    os.chdir(uwy_path)
    
    # if h5 file exists, load that, otherwise load from netCDF    
    if len(glob.glob('*.h5')) > 0:
        uwy_units = pd.read_hdf('uwy_units.h5',key='uwy_units')
        if filtOrRaw.lower() == 'filt':            
            if not timeResolution == '': # If a particular time resolution has been requested, then...
                # Check if particular time resolution file exists, if not create it, then load and return it
                fname = 'uwy_'+timeResolution.lower()+'.h5'
                if not os.path.isfile(fname):
                    resample_timebase(RawDataPath = uwy_path, input_h5_filename = 'uwy_filt', time_int=[timeResolution])
                uwy = pd.read_hdf(fname,key='uwy')
                return uwy
            else:
                uwy = pd.read_hdf('uwy_filt.h5',key='uwy')
                return (uwy, uwy_units)
        elif filtOrRaw.lower() == 'raw':
            uwy= pd.read_hdf('uwy_raw.h5',key='uwy')
        else:
            print('Please specify either the raw or filtered dataset as "raw" or "filt"')
    else:
        
        if (uwy_filename is None) or (~os.path.isfile(uwy_filename)): # if filename isn't specified, or that which is specified doesn't exist
            filename_list = glob.glob("*.nc")
            if len(filename_list) > 0:
                uwy_filename = filename_list[0]
            else:
                print("Please specify nc filename and ensure the file is in the specified directory")
                return        
        (uwy, uwy_units) = Load(uwy_path,'NC',NC_filename=uwy_filename)
        
    
    if filtOrRaw.lower() != 'filt':
       # Create Vector averaged wind direction
        if 'WindDirRel_vmean' not in uwy:
            if 'WindDirRel_port' in uwy:
                uwy = Create_WndDir_VectorMean(uwy)
        # QC Underway data
        uwy = UWY_QC(uwy, mask_period_timestamp_list = mask_period_timestamp_list)
        
        uwy.to_hdf('uwy_filt.h5',key='uwy')
    
    return (uwy, uwy_units)
    
#uwy_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/Underway/'
#
#UWY_LoadandProcess(uwy_path,filtOrRaw='raw')
            
