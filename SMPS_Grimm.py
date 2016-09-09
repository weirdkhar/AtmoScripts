def Load_to_HDF(RawDataPath, output_filename = 'SMPS_Grimm'):
	'''
	Function to load all raw files, concatenate and save as HDF file for easy loading
	'''
	import numpy as np
	import pandas as pd
	import glob
	import os
	
	os.chdir(RawDataPath)
	filename = output_filename
	
      # Get the list of files to import
	filelist = glob.glob('*.raw')
	filelist.sort()
 
      # Remove temp h5 file generated from the last time this was run
	if os.path.isfile(filename+'_temp.h5'):
		os.remove(filename+'_temp.h5')
	
      # Iterate through importing each file
	for i in range(len(filelist)):
		data_temp = pd.read_csv(filelist[i],
								header = 0,
								sep = '\t',
								engine = 'python')
		
		# Truncate the time to the nearest 5 minutes and make this the index
		ns5min=5*60*1000000000   # 5 minutes in nanoseconds
		time_index = pd.to_datetime(data_temp['Time Start'],dayfirst=True)
		time_index = pd.DatetimeIndex(((time_index.astype(np.int64) // ns5min + 1 ) * ns5min))
		
		data_temp['time_index'] = time_index
		data_temp = data_temp.set_index('time_index')
		
		# Initialise data variable, otherwise append data
		try:
			data
		except NameError:
			data = data_temp # data doesn't exist yet, initialise
		else:
			# data has been initialised, therefore, just append.
			data = data.append(data_temp)
		
		data.to_hdf(filename+'_temp.h5', key='smps')
		
	#Remove duplicates
	data = data.drop_duplicates()
 	
	# Sort data by ascending time
	data = data.sort_index()
	
      #Save the new dataset so you don't have to wait for the processing again.
	data.to_hdf(filename+'_raw.h5', key='smps')
     
     # Remove the temporary file
	os.remove(filename+'_temp.h5')
	return
	
def size_bins():
	import numpy as np
	bins = np.array([3.85,4.00,4.14,4.29,4.45,4.61,4.78,4.96,5.14,5.33,5.52,5.73,5.94,
			6.15,6.38,6.61,6.85,7.10,7.37,7.64,7.91,8.20,8.51,8.82,9.14,9.47,9.82,
			10.18,10.55,10.94,11.34,11.76,12.19,12.63,13.10,13.58,14.07,14.59,15.12,
			15.68,16.25,16.85,17.47,18.11,18.77,19.46,20.17,20.91,21.67,22.47,23.29,
			24.14,25.03,25.95,26.90,27.88,28.90,29.96,31.06,32.20,33.38,34.60,35.87,
			37.18,38.54,39.95,41.42,42.94,44.51,46.14,47.83,49.58,51.40,53.28,55.23,
			57.25,59.35,61.53,63.78,66.12,68.54,71.05,73.65,76.35,79.15,82.05,85.05,
			88.17,91.40,94.75,98.22,101.82,105.54,109.41,113.42,117.57,121.88,126.35,
			130.97,135.77,140.75,145.90,151.25,156.79,162.53,168.49,174.66,181.06,
			187.69,194.56,201.69,209.08,216.74,224.68,232.91,241.44,250.29,259.46,
			268.96,278.81,289.03,299.61,310.59,321.97,333.76,345.99,358.66,371.80,
			385.42,399.54,414.18,429.35])
	return bins

def plot_smps(data):
    '''
    Function to plot SMPS data from the grimm
    
    http://stackoverflow.com/questions/3716528/multi-panel-time-series-of-lines-and-filled-contours-using-matplotlib
    '''
    from matplotlib.colors import LogNorm
    import numpy as np
    import matplotlib.pyplot as plt
    # Check what form the data is in, raw, with all the parameter, or the cut down version
    if 'Liq' in data: # Raw form
        d_plt = data.iloc[:,4:136]
    else:
        # another form, need to figure out
        d_plt = 0
    x = data.index
    y = size_bins()
    
    X, Y = np.meshgrid(x,y)
    X = X.transpose()
    Y = Y.transpose()


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(X,Y,d_plt,levels=[1,1e1,1e2,1e3,1e4],cmap=plt.cm.plasma, norm = LogNorm())
    ax.set_yscale('log')
    plt.show()
    #plt.ylim([y.min(), y.max()])
    
    return

    
#import os
#import pandas as pd
#smpsgrimm_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/SMPS_GRIMM'
#os.chdir(smpsgrimm_path)
#s_raw = pd.read_hdf('SMPS_Grimm_raw.h5',key='smps')
#plot_smps(s_raw)