'''
Code for the gui of the processing of aerosol gear

Written by Ruhi Humphries
2017

Useful documentation: 
    http://www.tkdocs.com/tutorial/widgets.html 
    http://pyinmyeye.blogspot.com.au/2012/08/tkinter-combobox-demo.html
    http://www.python-course.eu/tkinter_layout_management.php
'''
import CCNC
import os
import sys
import tkinter as tk
#from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import threading

class ccn_processing(ttk.Frame):
    
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
            flow_cal_df = CCNC.load_flow_cals(file_FULLPATH=self.flowcal_file)
        except:
            flow_cal_df = None
            
        try:
            mask_df = CCNC.load_manual_mask(file_FULLPATH=self.mask_file)
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
        ''' UNCOMMENT WHEN FINISHED DEBUGGING
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
        '''
#######################
### When debugging, comment out to here.
### When finished, uncomment it so that the status window works.
####################### 

        # Call processing function
        CCNC.LoadAndProcess(
                ccn_raw_path = self.raw_path, 
                ccn_output_path = self.output_path,
                ccn_output_filetype = output_filetype,
                filename_base = 'CCN', 
                force_reload_from_source = self.forceReload.get(),
                split_by_supersaturation = self.split_SS.get(),
                QC = self.qc.get(), 
                output_time_resolution=output_time_res,
                concat_file_frequency = concat_file_freq,
                mask_period_timestamp_df = mask_df,
                flow_cal_df = flow_cal_df,
                calibrate_for_pressure = self.cb_pressCal,
                press_cal = float(self.tb_calPress.get()),
                press_meas = float(self.tb_measPress.get()),
                plot_each_step = self.plotresults.get(),
                input_filelist = list(self.files_raw)
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
        for i in range(0, len(self.files_raw)):
            self.lb_openFiles.insert(i,self.files_raw[i])
        try :
            self.output_path
        except AttributeError:
            self.t_outputPath.insert(tk.END,self.raw_path)
            self.output_path = self.raw_path
        return
        
    
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
        self.t_outputPath.delete(1.0,tk.END)
        self.t_outputPath.insert(tk.END,self.output_path)
                
    
    def grey_press_input(self):
        '''
        Disables input into the pressure fields if the checkbox isn't ticked.
        '''
        
        if self.correct4pressure.get() == 0:
            self.tb_calPress.configure(state='disabled')
            self.tb_measPress.configure(state='disabled')
        elif self.correct4pressure.get() == 1:
            self.tb_calPress.configure(state='normal')
            self.tb_measPress.configure(state='normal')
        
##-----------------------------------------------------------
## GUI Widgets
##-----------------------------------------------------------
    def __init__(self, isapp = True, name = 'ccnprocessing'):
        ttk.Frame.__init__(self, name=name)
        self.pack(expand=tk.Y, fill=tk.BOTH)
        self.master.title('DMT CCN Processing')
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
        self.cb_file_freq.current(0)  # set selection
        self.cb_file_freq.pack(pady=5, padx=10, side=tk.LEFT)
        self.cb_file_freq.place(rely=0.26, relx=0.375)

        # Create output supersaturation checkbox
        self.split_SS = tk.IntVar()
        self.cb_SS = tk.Checkbutton(self.f2,
                                    text = 'Split by supersaturation',
                                    variable=self.split_SS)
        self.cb_SS.select()
        self.cb_SS.pack(pady=5,padx=10)
        self.cb_SS.place(relx=0.02, rely=0.36)
        
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
        self.qc = tk.IntVar()
        self.cb_qc = tk.Checkbutton(self.f31, 
                               text="QC for internal parameters", 
                               variable=self.qc)
        self.cb_qc.select()
        self.cb_qc.pack(pady=5,padx=10)
        
        self.f311 = tk.LabelFrame(self.f31,
                    text='Select file with mask events (optional)'
                    )
        self.f311.pack(pady=5,padx=10, fill='x')
        self.tb2 = tk.Entry(self.f311, width=47) 
        self.tb2.pack(pady=5,padx=10, fill='x', side=tk.LEFT)
        self.b3 = tk.Button(self.f311,
                         text = "Browse",
                         command = self.ask_mask_file
                         ).pack(pady=5,padx=10, side=tk.LEFT)
        
        
        self.f32 = ttk.LabelFrame(self.f3, text='Data calibration')
        self.f32.pack(pady=5,padx=10, fill='x')
        
        self.f321 = ttk.LabelFrame(self.f32, text='Flow calibration')
        self.f321.pack(pady=5,padx=10, fill='x')
        self.lb3 = tk.Label(self.f321,
                    text='Select file with flow calibration data (optional)'
                    )
        self.lb3.pack(pady=5,padx=10, side=tk.TOP)
        
        self.tb3 = tk.Entry(self.f321, width=47) 
        self.tb3.pack(pady=5,padx=10, side=tk.LEFT)
        self.b3 = tk.Button(self.f321,
                         text = "Browse",
                         command = self.ask_flowcal_file
                         )
        self.b3.pack(pady=5,padx=10, side=tk.LEFT)
        
        
        self.f322 = ttk.LabelFrame(self.f32, text='Pressure calibration')
        self.f322.pack(pady=5,padx=10, fill='x')
        
        self.lb322 = tk.Label(self.f322, 
                      text = """Corrects reported supersaturation for changes \
in atmospheric pressure between calibration site and measurement site. If \
calibrated by DMT, calibration pressure is 830 hPa. Sea level pressure is 1010\
 hPa."""
                      ,wraplength=350,
                      )
        self.lb322.pack(pady=5,padx=10)
        
        self.correct4pressure = tk.IntVar()
        self.cb_pressCal = tk.Checkbutton(self.f322,
                                          text = 'Correct for pressure',
                                          variable = self.correct4pressure,
                                          onvalue = 1, offvalue = 0,
                                          command = self.grey_press_input)
        self.cb_pressCal.select()
        self.cb_pressCal.pack(pady=5, padx=10)
        
        self.f3221 = tk.LabelFrame(self.f322,text='Cal. Pressure')
        self.tb_calPress = tk.Entry(self.f3221, width = 5)
        self.tb_calPress.insert(tk.END,830)
        self.lb_units1 = tk.Label(self.f3221,text='hPa')
        
        self.f3221.pack(pady=5,padx=40, side=tk.LEFT, fill='x')
        self.tb_calPress.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_units1.pack(pady=5,padx=10, side=tk.LEFT)
        
        self.f3222 = tk.LabelFrame(self.f322,text='Meas. Pressure')
        self.tb_measPress = tk.Entry(self.f3222, width = 5)
        self.tb_measPress.insert(tk.END,1010)
        self.lb_units2 = tk.Label(self.f3222,text='hPa')
        
        self.f3222.pack(pady=5,padx=40, side=tk.RIGHT)
        self.tb_measPress.pack(pady=5,padx=10, side=tk.LEFT)
        self.lb_units2.pack(pady=5,padx=10, side=tk.RIGHT)
        
        
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
## Processing status window
##-----------------------------------------------------------
    def _build_status_window(self):
        self.w_status = tk.Toplevel()
        self.w_status.title('CCN Processing Status')
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
        self.w_finished.geometry("300x200")
        txt = tk.Message(self.w_finished,
                         text = "Processing completed!",
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
    ccn_processing().mainloop()
