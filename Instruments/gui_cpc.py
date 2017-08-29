'''
Code for the gui of the processing of aerosol gear

Written by Ruhi Humphries
2017

Useful documentation: 
    http://www.tkdocs.com/tutorial/widgets.html 
    http://pyinmyeye.blogspot.com.au/2012/08/tkinter-combobox-demo.html
    http://www.python-course.eu/tkinter_layout_management.php
'''
import sys
sys.path.append('h:\\code\\')
import CPC_TSI
import os
import sys
import tkinter as tk
#from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import threading
from AtmoScripts import atmoscripts
from AtmoScripts import ToolTip

class cpc_processing(ttk.Frame):
    
    def load_and_process(self):
        '''
        Once all parameters have been chosen, checks all the input values and
        begins the processing.
        '''

        # Initialise
        msg = 'There is an error with your input! \n'
        # Check input variables
        try:
            self.files_raw
        except AttributeError:
#        if self.files_raw is None:
            msg = msg + '\n Please select raw input files'
        
        try:
            self.output_path
            if not os.path.exists(self.output_path):
                msg = msg + '\n Chosen output path does not exist'
        except AttributeError:
            msg = msg + '\n Please select output path'
            
        
        if msg != 'There is an error with your input! \n':
            self._alert_bad_input(msg)
            return

        if self.cb_output_filetype.get() == 'netcdf': 
            if os.path.exists(self.output_path):
                os.chdir(self.output_path)
            
            atmoscripts.save_temp_glob_att(
                    self.nc_global_title,
                    self.nc_global_description,
                    self.nc_author,
                    self.nc_global_institution,
                    self.nc_global_comment)
            
        
        # Open new window showing status with the option to cancel execution 
        # and disable input window
        self._build_status_window()
        
        
        # Setup input
        output_time_res = [
                self.output_1s.get(),
                self.output_5s.get(),
                self.output_10s.get(),
                self.output_15s.get(),
                self.output_30s.get(),
                self.output_1m.get(),
                self.output_5m.get(),
                self.output_10m.get(),
                self.output_15m.get(),
                self.output_30m.get(),
                self.output_1h.get(),
                self.output_3h.get(),
                self.output_6h.get(),
                self.output_12h.get(),
                self.output_1d.get()
                ]
        # Change to boolean array
        output_time_res = [True if item == 1 
                           else False 
                           for item in output_time_res]
        
        if self.cb_file_freq.get() == 'Single file':
            concat_file_freq = 'all'
        elif self.cb_file_freq.get() == 'Daily files':
            concat_file_freq = 'daily'
        elif self.cb_file_freq.get() == 'Weekly files':
            concat_file_freq = 'weekly'
        elif self.cb_file_freq.get() == 'Monthly files':
            concat_file_freq = 'monthly'
        else:
            print('Something has gone terribly wrong here...')
            self.destroy()
        
        try:
            flow_cal_df = CPC_TSI.load_flow_cals(file_FULLPATH=self.flowcal_file)
        except:
            flow_cal_df = None
            
        try:
            mask_df = CPC_TSI.load_manual_mask(file_FULLPATH=self.mask_file)
        except:
            mask_df = None
        
        if self.cb_output_filetype.get() == 'netcdf':
            output_filetype = 'nc'
        elif self.cb_output_filetype.get() == 'hdf':
            output_filetype = 'h5'
        else:
            output_filetype = 'csv'
        
        print("Loading data from file")
        
#######################
### When debugging, comment out from here to the next break.
### When finished, uncomment it so that the status window works.
#######################        
#        ''' UNCOMMENT WHEN FINISHED DEBUGGING
        t = threading.Thread(target = self.loadAndProcess_Multithread,
                             args=(output_filetype,
                                   output_time_res,
                                   concat_file_freq,
                                   mask_df,
                                   flow_cal_df)
                             )
        t.start()        

    def loadAndProcess_Multithread(self,
                               output_filetype,
                               output_time_res, 
                               concat_file_freq,
                               mask_df,
                               flow_cal_df):
#        '''
#######################
### When debugging, comment out to here.
### When finished, uncomment it so that the status window works.
####################### 

        # Call processing function
        CPC_TSI.LoadAndProcess(        
                   cn_raw_path = self.raw_path, 
                   cn_output_path = self.output_path,
                   cn_output_filetype = output_filetype,
                   filename_base = 'CN3',
                   force_reload_from_source = self.forceReload.get(),
                   output_time_resolution = output_time_res,
                   concat_file_frequency = concat_file_freq,
                   input_filelist = list(self.files_raw),
                   NeedsTZCorrection = self.correct4TZ.get(),
                   CurrentTZ = float(self.tb_TZcurrent.get()), 
                   OutputTZ =  float(self.tb_TZdesired.get()),
                   mask_period_timestamp_df = mask_df,
                   flow_cal_df = flow_cal_df,
                   CN_flow_setpt = float(self.tb_flow_rate_set.get())*1000,
                   CN_flow_polyDeg = float(self.tb_flow_rate_fit.get()),
                   plot_each_step = self.plotresults.get(),
                   gui_mode = True,
                   gui_mainloop = self.w_status
                   )
        self.finished_window()

        
##-----------------------------------------------------------
## GUI Functionality
##-----------------------------------------------------------        
    def raw_file_dialog(self):
        '''Prompts user to select input files'''
        self.files_raw = filedialog.askopenfilenames()
        self.raw_path = os.path.dirname(self.files_raw[0])
        
        # Update the text box
        self.lb_openFiles.delete(0, tk.END)
        for i in range(0, len(self.files_raw)):
            self.lb_openFiles.insert(i,self.files_raw[i])
        try :
            if self.output_path == '':
                self.update_output_path()
        except AttributeError:
            self.update_output_path()
        return
    
    def update_output_path(self):
        self.t_outputPath.insert(tk.END,self.raw_path)
        self.output_path = self.raw_path
    
    def browse_for_file(self):
        '''Prompts user to select input file'''
        file = filedialog.askopenfilename()
        return file
    
    def ask_mask_file(self):
        ''' Asks for the mask file input and shows it in the gui'''
        self.mask_file = self.browse_for_file()
        self.tb2.insert(tk.END,self.mask_file)
        return
    
    def ask_flowcal_file(self):
        ''' Asks for the flow cal file input and shows it in the gui'''
        self.flowcal_file = self.browse_for_file()
        self.tb3.insert(tk.END,self.flowcal_file)
        return
    
    def output_path_dialog(self):
        '''Selecting output path, if not chosen, use the input directory'''
        self.output_path = filedialog.askdirectory()
        self.t_outputPath.delete(0,tk.END)
        self.t_outputPath.insert(tk.END,self.output_path)
                
    
    def grey_press_input(self):
        '''
        Disables input into the pressure fields if the checkbox isn't ticked.
        '''
        self.correct4TZ
        if self.correct4TZ.get() == 0:
            self.tb_TZcurrent.configure(state='disabled')
            self.tb_TZdesired.configure(state='disabled')
        elif self.correct4TZ.get() == 1:
            self.tb_TZcurrent.configure(state='normal')
            self.tb_TZdesired.configure(state='normal')
    
    def launch_netcdf_input(self, event):
        '''
        Launches netcdf input when the combobox option is selected
        '''
        if self.cb_output_filetype.get() == 'netcdf':
            self._build_netcdf_input_window()
        
    def close_netcdf_window(self):
        '''
        Closes the netcdf window on the OK button press and saves the input 
        as a temporary file which can be read by the code later.
        '''
        self.nc_global_title = self.nc_e0.get()
        self.nc_global_description = self.nc_e1.get()
        self.nc_author = self.nc_e2.get()
        self.nc_global_institution = self.nc_e3.get()
        self.nc_global_comment = self.nc_e4.get()
        self.w_netcdf_input.destroy()


##-----------------------------------------------------------
## GUI Widgets
##-----------------------------------------------------------
    def __init__(self, isapp = True, name = 'cpcprocessing'):
        ttk.Frame.__init__(self, name=name)
        self.pack(expand=tk.Y, fill=tk.BOTH)
        self.master.title('TSI CPC Processing')
        self.master.geometry("880x560")
        self.isapp = isapp
        self._build_widgets()

        
    def _build_widgets(self):
        global mainFrame
        mainFrame = tk.Frame(self)
        mainFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.Y)
        
        
        self._create_input_frame()
        self._create_output_frame()
        self._create_processing_frame()
        
        self.f1.pack(in_=mainFrame, side=tk.TOP, pady=5, padx=10)
        self.f2.pack(in_=mainFrame, side=tk.TOP, pady=5, padx=10)
        self.f3.pack(in_=mainFrame, side=tk.TOP, pady=5, padx=10)
        self.f1.place(relx=0.01,rely=0.01,relheight=0.39,relwidth=0.49)
        self.f2.place(relx=0.01,rely=0.4,relheight=0.6,relwidth=0.49)
        self.f3.place(relx=0.51,rely=0.01,relheight=1,relwidth=0.49)
        
    def _create_input_frame(self):
        self.f1 = ttk.LabelFrame(mainFrame, text = 'Input data')
        
        # Create open file dialog input
        self.b_open = tk.Button(self.f1,
                       text = "Select raw files",
                       command = self.raw_file_dialog
                       )
        self.b_open.pack(pady=5,padx=10,side=tk.TOP)
        self.b_open.place(relx=0.02,rely=0.02)
        
        self.f11 = ttk.Frame(self.f1)
        self.f11.pack(pady=5,padx=10,side=tk.LEFT)
        self.f11.place(relx=0.02,rely=0.15, relheight=0.83, relwidth=0.96)
        
        self.lb_openFiles = tk.Listbox(self.f11)
        self.sb_openFiles = tk.Scrollbar(self.f11)
        
        self.lb_openFiles.pack(side=tk.LEFT, fill='both',expand=True)
        self.sb_openFiles.pack(side=tk.LEFT, fill='y')
        
        # Attach listbox to scrollbar
        self.lb_openFiles.config(yscrollcommand=self.sb_openFiles.set)
        self.sb_openFiles.config(command=self.lb_openFiles.yview)
        
        
        # Create forceReload check button
        self.forceReload = tk.IntVar()
        self.cb_forceReload = tk.Checkbutton(self.f1,
                                        text="Force reload from source",
                                        variable = self.forceReload)
        self.cb_forceReload.select()
        self.cb_forceReload.pack(pady=5,padx=10,side=tk.TOP)
        self.cb_forceReload.place(relx=0.52,rely=0.02)
        
        
    def _create_output_frame(self):
        self.f2 = ttk.LabelFrame(mainFrame, text = 'Output data')
        
        
        # create output path dialog
        self.b_output = tk.Button(self.f2,
                         text = "Change output directory",
                         command = self.output_path_dialog
                         )
        self.b_output.pack(pady=5,padx=10, side=tk.LEFT)
        self.b_output.place(rely=0.05,relx=0.02)
        self.t_outputPath = tk.Entry(self.f2, width=42)
        self.t_outputPath.pack(pady=5,padx=10, side=tk.LEFT)
        self.t_outputPath.place(rely=0.06, relx=0.375)
        
        # Create output filetype combobox
        filetypes = ['netcdf','hdf','csv']
        self.lb1 = ttk.Label(self.f2,
                             text = 'Select output filetype')
        self.lb1.pack(pady=5,padx=10,side=tk.LEFT)
        self.lb1.place(rely=0.16, relx=0.02)
        
        self.cb_output_filetype = ttk.Combobox(self.f2, 
                                values=filetypes, 
                                state='readonly', 
                                width = 10)
        self.cb_output_filetype.current(1)  # set selection
        self.cb_output_filetype.pack(pady=5, padx=10, side=tk.LEFT)
        self.cb_output_filetype.place(rely=0.16, relx=0.375)
        self.cb_output_filetype.bind("<<ComboboxSelected>>",self.launch_netcdf_input)
        
        # Create output file frequency combobox
        file_freq=['Single file','Daily files','Weekly files','Monthly files']
        self.lb2 = tk.Label(self.f2, 
                            text = 'Select output frequency')
        self.lb2.pack(pady=5,padx=10,side=tk.LEFT)
        self.lb2.place(rely=0.26, relx=0.02)
        
        self.cb_file_freq = ttk.Combobox(self.f2, 
                                values=file_freq, 
                                state='readonly', 
                                width = 15)
        self.cb_file_freq.current(2)  # set selection
        self.cb_file_freq.pack(pady=5, padx=10, side=tk.LEFT)
        self.cb_file_freq.place(rely=0.26, relx=0.375)

#==============================================================================
#         # Create output supersaturation checkbox
#         self.split_SS = tk.IntVar()
#         self.cb_SS = tk.Checkbutton(self.f2,
#                                     text = 'Split by supersaturation',
#                                     variable=self.split_SS)
#         self.cb_SS.select()
#         self.cb_SS.pack(pady=5,padx=10)
#         self.cb_SS.place(relx=0.02, rely=0.36)
#==============================================================================
        
        # Create output time resolution options
        self.f21 = ttk.LabelFrame(self.f2,
                             text='Output time resolution')
        self.f21.pack(pady=5,padx=10, fill='x')
        self.f21.place(rely=0.46, relx=0.02, relwidth=0.96, relheight=0.50)
        
        # Declare checkbox variables
        self.output_1s = tk.IntVar()
        self.output_5s = tk.IntVar()
        self.output_10s = tk.IntVar()
        self.output_15s = tk.IntVar()
        self.output_30s = tk.IntVar()
        self.output_1m = tk.IntVar()
        self.output_5m = tk.IntVar()
        self.output_10m = tk.IntVar()
        self.output_15m = tk.IntVar()
        self.output_30m = tk.IntVar()
        self.output_1h = tk.IntVar()
        self.output_3h = tk.IntVar()
        self.output_6h = tk.IntVar()
        self.output_12h = tk.IntVar()
        self.output_1d = tk.IntVar()
        
        # Create checkboxes
        self.cb_1s = tk.Checkbutton(self.f21, text="1 second", variable=self.output_1s)
        self.cb_5s = tk.Checkbutton(self.f21, text="5 seconds", variable=self.output_5s)
        self.cb_10s = tk.Checkbutton(self.f21, text="10 seconds", variable=self.output_10s)
        self.cb_15s = tk.Checkbutton(self.f21, text="15 seconds", variable=self.output_15s)
        self.cb_30s = tk.Checkbutton(self.f21, text="30 seconds", variable=self.output_30s)
        self.cb_1m = tk.Checkbutton(self.f21, text="1 minute", variable=self.output_1m)
        self.cb_5m = tk.Checkbutton(self.f21, text="5 minutes", variable=self.output_5m)
        self.cb_10m = tk.Checkbutton(self.f21, text="10 minutes", variable=self.output_10m)
        self.cb_15m = tk.Checkbutton(self.f21, text="15 minutes", variable=self.output_15m)
        self.cb_30m = tk.Checkbutton(self.f21, text="30 minutes", variable=self.output_30m)
        self.cb_1h = tk.Checkbutton(self.f21, text="1 hour", variable=self.output_1h)
        self.cb_3h = tk.Checkbutton(self.f21, text="3 hours", variable=self.output_3h)
        self.cb_6h = tk.Checkbutton(self.f21, text="6 hours", variable=self.output_6h)
        self.cb_12h = tk.Checkbutton(self.f21, text="12 hours", variable=self.output_12h)
        self.cb_1d = tk.Checkbutton(self.f21, text="1 day", variable=self.output_1d)		
        
        self.cb_1s.select() # Select default value as checked
        
        # Position
        self.cb_1s.pack(pady=2,padx=10)
        self.cb_5s.pack(pady=2,padx=10)
        self.cb_10s.pack(pady=2,padx=10)
        self.cb_15s.pack(pady=2,padx=10)
        self.cb_30s.pack(pady=2,padx=10)
        self.cb_1m.pack(pady=2,padx=10)
        self.cb_5m.pack(pady=2,padx=10)
        self.cb_10m.pack(pady=2,padx=10)
        self.cb_15m.pack(pady=2,padx=10)
        self.cb_30m.pack(pady=2,padx=10)
        self.cb_1h.pack(pady=2,padx=10)
        self.cb_3h.pack(pady=2,padx=10)
        self.cb_6h.pack(pady=2,padx=10)
        self.cb_12h.pack(pady=2,padx=10)
        self.cb_1d.pack(pady=2,padx=10)
        
        self.cb_1s.place(relx=0.02, rely=0.02)
        self.cb_5s.place(relx=0.02, rely=0.20)
        self.cb_10s.place(relx=0.02,rely=0.38)
        self.cb_15s.place(relx=0.02,rely=0.56)
        self.cb_30s.place(relx=0.02,rely=0.74)
        
        self.cb_1m.place(relx=0.33, rely=0.02)
        self.cb_5m.place(relx=0.33, rely=0.20)
        self.cb_10m.place(relx=0.33,rely=0.38)
        self.cb_15m.place(relx=0.33,rely=0.56)
        self.cb_30m.place(relx=0.33,rely=0.74)
        
        self.cb_1h.place(relx=0.67, rely=0.02)
        self.cb_3h.place(relx=0.67, rely=0.20)       
        self.cb_6h.place(relx=0.67, rely=0.38)
        self.cb_12h.place(relx=0.67,rely=0.56)
        self.cb_1d.place(relx=0.67, rely=0.74)
        
        
        
    def _create_processing_frame(self):
        self.f3 = ttk.LabelFrame(mainFrame, text = 'Processing options')
        
        # Data mask/removal frame
        self.f31 = ttk.LabelFrame(self.f3, text='Data masking/removal')
        self.f31.pack(pady=5,padx=10, fill='x')
#        self.qc = tk.IntVar()
#        self.cb_qc = tk.Checkbutton(self.f31, 
#                               text="QC for internal parameters", 
#                               variable=self.qc)
#        self.cb_qc.select()
#        self.cb_qc.pack(pady=5,padx=10)
        
        self.f311 = tk.LabelFrame(self.f31,
                    text='Select file with mask events (optional)'
                    )
        self.f311.pack(pady=5,padx=10, fill='x')
        self.tb2 = tk.Entry(self.f311, width=45) 
        self.tb2.pack(pady=5,padx=10, fill='x', side=tk.LEFT)
        self.b3 = tk.Button(self.f311,
                         text = "Browse",
                         command = self.ask_mask_file
                         ).pack(pady=5,padx=10, side=tk.LEFT)
        
        
        self.f32 = ttk.LabelFrame(self.f3, text='Flow calibration')
        self.f32.pack(pady=5,padx=10, fill='x')
        
        # Create help tooltip
        self.l311 = tk.Label(self.f311,text = u'\u2754')
        self.l311.pack(pady=5,side=tk.LEFT)
        ToolTip.ToolTip(self.l311,
                        "Choose an ASCII file where the 1st and 2nd columns \
                        are the start and end timestamps of the period to be \
                        removed. Any additional columns (such as description \
                        columns) will be ignored.")
        
        
        self.f321 = tk.LabelFrame(self.f32, 
                    text='Select file with flow calibration data (optional)')
        self.f321.pack(pady=5,padx=10, fill='x')
        
        self.tb3 = tk.Entry(self.f321, width=45) 
        self.tb3.pack(pady=5,padx=10, side=tk.LEFT)
        self.b3 = tk.Button(self.f321,
                         text = "Browse",
                         command = self.ask_flowcal_file
                         )
        self.b3.pack(pady=5,padx=10, side=tk.LEFT)
        
        # Create help tooltip
        self.l321 = tk.Label(self.f321,text = u'\u2754')
        self.l321.pack(pady=5,side=tk.LEFT)
        ToolTip.ToolTip(self.l321,
                        "Choose an ASCII file where the 1st column is the  \
                        timestamp of the flow measurement, and the second \
                        column is the measured flow rate in units of \
                        L/min or LPM")
        
        self.lb_flow_rate_set = tk.Label(self.f32,text="Set flow rate (LPM)")
        self.tb_flow_rate_set = tk.Entry(self.f32, width = 10)
        self.tb_flow_rate_set.insert(tk.END,1.0)
        self.lb_flow_rate_set.pack(pady=5,padx=10, side=tk.LEFT)
        self.tb_flow_rate_set.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_flow_rate_set.place(relx=0.02, rely=0.55)
        self.tb_flow_rate_set.place(relx=0.52, rely=0.55)
        
        
        self.lb_flow_rate_fit = tk.Label(self.f32,
                                    text="Polynomial degree for flow rate fit")
        self.tb_flow_rate_fit = tk.Entry(self.f32, width = 10)
        self.tb_flow_rate_fit.insert(tk.END,2)
        self.lb_flow_rate_fit.pack(pady=5,padx=10, side=tk.LEFT)
        self.tb_flow_rate_fit.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_flow_rate_fit.place(relx=0.02, rely=0.8)
        self.tb_flow_rate_fit.place(relx=0.52, rely=0.8)
        
        
        
#==============================================================================
        self.f322 = ttk.LabelFrame(self.f3, text='Time Zone Correction')
        self.f322.pack(pady=5,padx=10, fill='x')
         
        self.lb322 = tk.Label(self.f322, 
                       text = "Corrects timestamp for offset created by AIM \
outputting the timestamps based on the export computer's settings, rather \
than the measurement computer's time zone settings."
                       ,wraplength=350,
                       )
        self.lb322.pack(pady=5,padx=10)
         
        self.correct4TZ = tk.IntVar()
        self.cb_TZcorrection = tk.Checkbutton(self.f322,
                                           text = 'Correct Time Zone',
                                           variable = self.correct4TZ,
                                           onvalue = 1, offvalue = 0,
                                           command = self.grey_press_input)
        self.cb_TZcorrection.pack(pady=5, padx=10)
         
        self.f3221 = tk.LabelFrame(self.f322,
                                    text="Export PC's TZ (current)")
        self.tb_TZcurrent = tk.Entry(self.f3221, width = 5)
        self.tb_TZcurrent.insert(tk.END,0)
        self.lb_units1 = tk.Label(self.f3221,text='hrs from UTC')
         
        self.f3221.pack(pady=5,padx=10, side=tk.LEFT, fill='x')
        self.tb_TZcurrent.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_units1.pack(pady=5,padx=10, side=tk.LEFT)
         
        self.f3222 = tk.LabelFrame(self.f322
                                    ,text="Meas. PC's TZ (desired)")
        self.tb_TZdesired = tk.Entry(self.f3222, width = 5)
        self.tb_TZdesired.insert(tk.END,0)
        self.lb_units2 = tk.Label(self.f3222,text='hrs from UTC')
         
        self.f3222.pack(pady=5,padx=10, side=tk.RIGHT)
        self.tb_TZdesired.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_units2.pack(pady=5,padx=10, side=tk.RIGHT)
#==============================================================================
        
        
        self.plotresults = tk.IntVar()
        self.cb_plot = tk.Checkbutton(self.f3,
                                      text = 'Plot after each step',
                                      variable = self.plotresults,
                                      onvalue = True, offvalue = False)
        self.cb_plot.pack()
        
        # Create go button!
        self.bt_go = tk.Button(self.f3,
                       text='GO!',
                       command=self.load_and_process,
                       background='forest green',
                       foreground='white',
                       font = 'Times 18 bold',
                       width = 15)
        self.bt_go.pack(side=tk.BOTTOM)
        self.bt_go.place(rely=0.89, relx=0.25)

        self.f31.place(relx=0.01,rely=0.04,relheight=0.15,relwidth=0.98)
        self.f32.place(relx=0.01,rely=0.2,relheight=0.26,relwidth=0.98)
        self.f322.place(relx=0.01,rely=0.47,relheight=0.33,relwidth=0.98)
        self.cb_plot.place(relx=0.3,rely=0.83)
##-----------------------------------------------------------
## Variable check window
##-----------------------------------------------------------
    def _alert_bad_input(self, message='Nothing to see here...'):
        self.top = tk.Toplevel()
        self.top.title('Bad input!')
        self.top.geometry("%dx%d" % (300, 200))
        txt = tk.Message(self.top, 
                         text=message, 
                         justify=tk.CENTER,
                         width = 300)
        txt.pack(fill='x')
        
        bt_ok = tk.Button(self.top,
                          text="OK",
                          command=self.dismiss
                          )
        bt_ok.pack(side=tk.BOTTOM)
        
    def dismiss(self):
        self.top.destroy()

##-----------------------------------------------------------
## NetCDF Description input window
##-----------------------------------------------------------
    def _build_netcdf_input_window(self):
        self.w_netcdf_input = tk.Toplevel()
        self.w_netcdf_input.title('NetCDF Input')
        self.w_netcdf_input.geometry("300x310")
        
        self.w_netcdf_input.description = tk.Label(self.w_netcdf_input,
                text = 'Please provide descriptions (global attributes) to be \
                included in the self-describing NetCDF file \n', 
                wraplength=300)
        self.w_netcdf_input.description.pack()
        
        text = 'Dataset title'
        self.nc_l0 = tk.Label(self.w_netcdf_input,
                                          text = text,
                                          wraplength = 200)
        self.nc_l0.pack()
        self.nc_e0 = tk.Entry(self.w_netcdf_input,width = 200)
        self.nc_e0.pack()
        
        text = 'Dataset description'
        self.nc_l1 = tk.Label(self.w_netcdf_input,
                                          text = text,
                                          wraplength = 200)
        self.nc_l1.pack()
        self.nc_e1 = tk.Entry(self.w_netcdf_input,width = 200)
        self.nc_e1.pack()
        
        text = 'Author of dataset'
        self.nc_l2 = tk.Label(self.w_netcdf_input,
                                          text = text,
                                          wraplength = 200)
        self.nc_l2.pack()
        self.nc_e2 = tk.Entry(self.w_netcdf_input,width = 200)
        self.nc_e2.pack()
        
        text = 'Institution where dataset is produced'
        self.nc_l3 = tk.Label(self.w_netcdf_input,
                                          text = text,
                                          wraplength = 200)
        self.nc_l3.pack()
        self.nc_e3 = tk.Entry(self.w_netcdf_input,width = 200)
        self.nc_e3.pack()
        
        text = 'Comment'
        self.nc_l4 = tk.Label(self.w_netcdf_input,
                                          text = text,
                                          wraplength = 200)
        self.nc_l4.pack()
        self.nc_e4 = tk.Entry(self.w_netcdf_input,width = 50)
        self.nc_e4.pack()
        
        self.w_netcdf_input.spacer = tk.Label(self.w_netcdf_input,
                                          text = ''
                                          ).pack()
        
        self.nc_bt_ok = tk.Button(self.w_netcdf_input,
                          text="OK",
                          width = 30,
                          command=self.close_netcdf_window)
        self.nc_bt_ok.pack()
  

        
##-----------------------------------------------------------
## Processing status window
##-----------------------------------------------------------
    def _build_status_window(self):
        self.w_status = tk.Toplevel()
        self.w_status.title('CPC Processing Status')
        self.w_status.geometry("800x500")
        
        self.w_status.txt_status = tk.Text(self.w_status, wrap='word')
        self.w_status.sb_status = tk.Scrollbar(self.w_status)
        self.w_status.txt_status.pack(pady=5, side=tk.LEFT,fill='both', expand=True)    
        self.w_status.sb_status.pack(pady=5,side=tk.LEFT, fill='y')
        
        # Attach listbox to scrollbar
        self.w_status.txt_status.config(yscrollcommand=self.w_status.sb_status.set)
        self.w_status.sb_status.config(command=self.w_status.txt_status.yview)
       
#        bt_interupt = tk.Button(self.w_status,
#                             text='Interupt', 
#                             command=self.interupt,
#                             bg='red',
#                             fg='white'
#                             )
#        bt_interupt.pack(pady=10, side=tk.BOTTOM)
        
        self.w_status.txt_status.tag_configure("stderr", foreground="#b22222")
        sys.stdout = TextRedirector(self.w_status.txt_status,"stdout")
        sys.stderr = TextRedirector(self.w_status.txt_status,"stderr")
       
#    def interupt(self):
#        '''
#        Stops the execution 
#        '''
#        self.w_interupt_check = tk.Toplevel()
#        self.w_interupt_check.title('Cancel processing')
#        
#        l1 = tk.Label(self.w_interupt_check,
#                      text='Are you sure you want to exit?') 
#        l2 = tk.Label(self.w_interupt_check,
#                      text="""This will exit the program and you will have 
#to launch the program again"""
#                      )
#        l1.pack()
#        l2.pack()
#        
#        bt_y = tk.Button(self.w_interupt_check,
#                      text="Yes, get me out of here!",
#                      command=self.interupt_yes,
#                      bg='red',
#                      fg='white')
#        bt_n = tk.Button(self.w_interupt_check,
#                      text="No, please continue",
#                      command=self.interupt_no,
#                      bg='green',
#                      fg='white')
#        bt_y.pack(side=tk.LEFT,padx=60,pady=5)
#        bt_n.pack(side=tk.LEFT,padx=60,pady=5)
#        
#        
#    def interupt_yes(self):
#        sys.exit(0)
#        
#    def interupt_no(self):
#        self.w_interupt_check.destroy()


    def finished_window(self):
        self.w_finished = tk.Toplevel()
        self.w_finished.title("All finished")
        self.w_finished.geometry("300x100")
        txt = tk.Message(self.w_finished,
                         text = "Processing of CPC data complete!",
                         justify=tk.CENTER,
                         width=300)
        txt.pack()
        bt_ok = tk.Button(self.w_finished,
                          text="OK",
                          command=self.finish)
        bt_ok.pack()
        
    def finish(self):
        self.w_finished.destroy()
        self.w_status.destroy()
        
class TextRedirector(object):
    # Taken from 
    # http://stackoverflow.com/questions/12351786/python-converting-cli-to-gui
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see(tk.END) # Scroll with text
        
if __name__ == '__main__':
    cpc_processing().mainloop()
