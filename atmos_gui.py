'''
Code for the gui of the processing of aerosol gear

Written by Ruhi Humphries
2017

Useful documentation: 
    http://www.tkdocs.com/tutorial/widgets.html 
    http://pyinmyeye.blogspot.com.au/2012/08/tkinter-combobox-demo.html
    http://www.python-course.eu/tkinter_layout_management.php
'''
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
global files_raw, default_output_path



class ccn_processing(ttk.Frame):
    
        
    def open_file_dialog(self):
        '''
        Prompts user to select input files
        '''
        files_raw = filedialog.askopenfilenames()
        output_path_default = os.path.dirname(files_raw[0])
        
        for i in range(0, len(files_raw)):
            lb_openFiles.insert(i,files_raw[i])
        if output_path is None:
            t_outputPath.insert(END,output_path_default)
            output_path = output_path_default
    
    def browse_for_file(self):
        '''
        Prompts user to select input file
        '''
        file = filedialog.askopenfilename()
        return file
    
    def open_path_dialog():
        '''
        Selecting output path, if not chosen, use the input directory
        '''
        global output_path
        output_path = filedialog.askdirectory()
        t_outputPath.delete(1.0,END)
        t_outputPath.insert(END,output_path)
        
        
    def reload_from_source():
        '''
        allows user to force reload from source files
        '''        
            
    def __init__(self, isapp = True, name = 'ccnprocessing'):
        ttk.Frame.__init__(self, name=name)
        self.pack(expand=Y, fill=BOTH)
        self.master.title('DMT CCN Processing')
        self.master.geometry("880x560")
        self.isapp = isapp
        self._build_widgets()
        
    def _build_widgets(self):
        global mainFrame
        mainFrame = Frame(self)
        mainFrame.pack(side=TOP, fill=BOTH, expand=Y)
        
        
        self._create_input_frame()
        self._create_output_frame()
        self._create_processing_frame()
        
        f1.pack(in_=mainFrame, side=TOP, pady=5, padx=10)
        f2.pack(in_=mainFrame, side=TOP, pady=5, padx=10)
        f3.pack(in_=mainFrame, side=TOP, pady=5, padx=10)
        f1.place(relx=0.01,rely=0.01,relheight=0.49,relwidth=0.49)
        f2.place(relx=0.01,rely=0.5,relheight=0.5,relwidth=0.49)
        f3.place(relx=0.51,rely=0.01,relheight=1,relwidth=0.49)
        
    def _create_input_frame(self):
        global f1
        f1 = ttk.LabelFrame(mainFrame, text = 'Input data')
        
        # Create open file dialog input
        b_open = tk.Button(f1,
                       text = "Select raw files",
                       command = self.open_file_dialog
                       )
        b_open.pack(pady=5,padx=10,side=TOP)
        b_open.place(relx=0.02,rely=0.02)
        
        f11 = ttk.Frame(f1)
        f11.pack(pady=5,padx=10,side=LEFT)
        f11.place(relx=0.02,rely=0.15, relheight=0.83, relwidth=0.96)
        
        global lb_openFiles
        lb_openFiles = tk.Listbox(f11)
        sb_openFiles = tk.Scrollbar(f11)
        
        lb_openFiles.pack(side=LEFT, fill='both',expand=True)
        sb_openFiles.pack(side=LEFT, fill='y')
        
        # Attach listbox to scrollbar
        lb_openFiles.config(yscrollcommand=sb_openFiles.set)
        sb_openFiles.config(command=lb_openFiles.yview)
        
        
        # Create forceReload check button
        forceReload = IntVar()
        cb_forceReload = tk.Checkbutton(f1,
                                        text="Force reload from source",
                                        variable = forceReload)
        cb_forceReload.pack(pady=5,padx=10,side=TOP)
        cb_forceReload.place(relx=0.52,rely=0.02)
        
    def _create_output_frame(self):
        global f2
        f2 = ttk.LabelFrame(mainFrame, text = 'Output data')
        
        
        # create output path dialog
        b_output = tk.Button(f2,
                         text = "Change output directory",
                         command = self.open_path_dialog
                         )
        b_output.pack(pady=5,padx=10, side=LEFT)
        b_output.place(rely=0.05,relx=0.02)
        t_outputPath = tk.Entry(f2, width=42)
        t_outputPath.pack(pady=5,padx=10, side=LEFT)
        t_outputPath.place(rely=0.06, relx=0.375)
        
        # Create output filetype combobox
        filetypes = ['netcdf','hdf','csv']
        lb1 = ttk.Label(f2, text = 'Select output filetype'
                        )
        lb1.pack(pady=5,padx=10,side=LEFT)
        lb1.place(rely=0.25, relx=0.02)
        cb1 = ttk.Combobox(f2, values=filetypes, state='readonly', width = 10)
        cb1.current(0)  # set selection
        cb1.pack(pady=5, padx=10, side=LEFT)
        cb1.place(rely=0.25, relx=0.375)
        
        # Create output time resolution options
        f21 = ttk.LabelFrame(f2,text='Output time resolution')
        f21.pack(pady=5,padx=10, fill='x')
        f21.place(rely=0.4, relx=0.02, relwidth=0.96, relheight=0.58)

        # Declare checkbox variables
        output_1s = IntVar
        output_5s = IntVar
        output_10s = IntVar
        output_15s = IntVar
        output_30s = IntVar
        output_1m = IntVar
        output_5m = IntVar
        output_10m = IntVar
        output_15m = IntVar
        output_30m = IntVar
        output_1h = IntVar
        output_3h = IntVar
        output_6h = IntVar
        output_12h = IntVar
        output_1d = IntVar
        
        # Create checkboxes
        cb_1s = tk.Checkbutton(f21, text="1 second", variable=output_1s)
        cb_5s = tk.Checkbutton(f21, text="5 seconds", variable=output_5s)
        cb_10s = tk.Checkbutton(f21, text="10 seconds", variable=output_10s)
        cb_15s = tk.Checkbutton(f21, text="15 seconds", variable=output_15s)
        cb_30s = tk.Checkbutton(f21, text="30 seconds", variable=output_30s)
        cb_1m = tk.Checkbutton(f21, text="1 minute", variable=output_1m)
        cb_5m = tk.Checkbutton(f21, text="5 minutes", variable=output_5m)
        cb_10m = tk.Checkbutton(f21, text="10 minutes", variable=output_10m)
        cb_15m = tk.Checkbutton(f21, text="15 minutes", variable=output_15m)
        cb_30m = tk.Checkbutton(f21, text="30 minutes", variable=output_30m)
        cb_1h = tk.Checkbutton(f21, text="1 hour", variable=output_1h)
        cb_3h = tk.Checkbutton(f21, text="3 hours", variable=output_3h)
        cb_6h = tk.Checkbutton(f21, text="6 hours", variable=output_6h)
        cb_12h = tk.Checkbutton(f21, text="12 hours", variable=output_12h)
        cb_1d = tk.Checkbutton(f21, text="1 day", variable=output_1d)		
        
        cb_1s.select() # Select default value as checked
        
        # Position
        cb_1s.pack(pady=5,padx=10)
        cb_5s.pack(pady=5,padx=10)
        cb_10s.pack(pady=5,padx=10)
        cb_15s.pack(pady=5,padx=10)
        cb_30s.pack(pady=5,padx=10)
        cb_1m.pack(pady=5,padx=10)
        cb_5m.pack(pady=5,padx=10)
        cb_10m.pack(pady=5,padx=10)
        cb_15m.pack(pady=5,padx=10)
        cb_30m.pack(pady=5,padx=10)
        cb_1h.pack(pady=5,padx=10)
        cb_3h.pack(pady=5,padx=10)
        cb_6h.pack(pady=5,padx=10)
        cb_12h.pack(pady=5,padx=10)
        cb_1d.pack(pady=5,padx=10)
        
        cb_1s.place(relx=0.02, rely=0.02)
        cb_5s.place(relx=0.02, rely=0.22)
        cb_10s.place(relx=0.02,rely=0.42)
        cb_15s.place(relx=0.02,rely=0.62)
        cb_30s.place(relx=0.02,rely=0.82)
        
        cb_1m.place(relx=0.33, rely=0.02)
        cb_5m.place(relx=0.33, rely=0.22)
        cb_10m.place(relx=0.33,rely=0.42)
        cb_15m.place(relx=0.33,rely=0.62)
        cb_30m.place(relx=0.33,rely=0.82)
        
        cb_1h.place(relx=0.67, rely=0.02)
        cb_3h.place(relx=0.67, rely=0.22)       
        cb_6h.place(relx=0.67, rely=0.42)
        cb_12h.place(relx=0.67,rely=0.62)
        cb_1d.place(relx=0.67, rely=0.82)
        
        
    def _create_processing_frame(self):
        global f3    
        f3 = ttk.LabelFrame(mainFrame, text = 'Processing options')
        
        # Data mask/removal frame
        f31 = ttk.LabelFrame(f3, text='Data masking/removal')
        f31.pack(pady=5,padx=10, fill='x')
        qc = IntVar
        #qc.set(value=1)
        cb_qc = tk.Checkbutton(f31, 
                               text="QC for internal parameters", 
                               variable=qc)
        cb_qc.select()
        cb_qc.pack(pady=5,padx=10)
        
        lb2 = Label(f31,
                    text='Select file with mask events (optional)'
                    ).pack(pady=5,padx=10)
        tb2 = Entry(f31)
        tb2.pack(pady=5,padx=10, fill='x')
        b3 = tk.Button(f31,
                         text = "Browse",
                         command = self.browse_for_file
                         ).pack(pady=5,padx=10)
        
        
        f32 = ttk.LabelFrame(f3, text='Data calibration')
        f32.pack(pady=5,padx=10, fill='x')
        
        f321 = ttk.LabelFrame(f32, text='Flow calibration')
        f321.pack(pady=5,padx=10, fill='x')
        lb3 = Label(f321,
                    text='Select file with flow calibration data (optional)'
                    ).pack(pady=5,padx=10)
        
        tb3 = Entry(f321)
        tb3.pack(pady=5,padx=10)
        b3 = tk.Button(f321,
                         text = "Browse",
                         command = self.browse_for_file
                         ).pack(pady=5,padx=10)
        
        
        f322 = ttk.LabelFrame(f32, text='Pressure calibration')
        f322.pack(pady=5,padx=10, fill='x')
        
        lb322 = Label(f322, 
                      text = """Corrects reported supersaturation for changes \
in atmospheric pressure between calibration site and measurement site. If \
calibrated by DMT, calibration pressure is 830 hPa. Sea level pressure is 1010\
 hPa."""
                      ,wraplength=350,
                      )
        lb322.pack(pady=5,padx=10)
        
        f3221 = LabelFrame(f322,text='Calibration Pressure')
        tb_calPress = Entry(f3221, width = 5)
        tb_calPress.insert(END,830)
        lb_units1 = Label(f3221,text='hPa')
        
        f3221.pack(pady=5,padx=40, side=LEFT, fill='x')
        tb_calPress.pack(pady=5,padx=10, side=LEFT)
        lb_units1.pack(pady=5,padx=10, side=LEFT)
        
        f3222 = LabelFrame(f322,text='Measurement Pressure')
        tb_measPress = Entry(f3222, width = 5)
        tb_measPress.insert(END,1010)
        lb_units2 = Label(f3222,text='hPa')
        
        f3222.pack(pady=5,padx=40, side=RIGHT)
        tb_measPress.pack(pady=5,padx=10, side=LEFT)
        lb_units2.pack(pady=5,padx=10, side= RIGHT)
        

    
    
if __name__ == '__main__':
    ccn_processing().mainloop()

'''
def main():
    root = tk.Tk()
    root.geometry("480x460")
    root.title("Aerosol Microphysics Processing")
    
    # Selecting raw files to concatenate
    def open_file_dialog():
        files_raw = filedialog.askopenfilenames()
        output_path_default = os.path.dirname(files_raw[0])
        
        for i in range(0, len(files_raw)):
            lb_openFiles.insert(i,files_raw[i])
        if output_path is None:
            t_outputPath.insert(END,output_path_default)
            output_path = output_path_default
        

    
    b_open = tk.Button(root,
                       text = "Select raw files",
                       command = open_file_dialog
                       ).pack()
                       #).grid(row=0,column=0)
    
    lb_openFiles = tk.Listbox(root)
    lb_openFiles.pack()
    
    # Selecting output path, if not chosen, use the input directory
    def open_path_dialog():
        output_path = filedialog.askdirectory()
        global output_path
        t_outputPath.delete(1.0,END)
        t_outputPath.insert(END,output_path)
    
    b_output = tk.Button(root,
                         text = "Change output directory",
                         command = open_path_dialog
                         ).pack()
    t_outputPath = tk.Text(root)
    t_outputPath.pack()
    
    # Choose output filetype
    
    
    
    filetypes = ['netcdf','hdf','csv']
    cb_filetype = ttk.Combobox(root, values=filetypes, state='readonly')
    cb_filetype.current(0) # set selection
    cb_filetype.pack()
    
    root.mainloop()
    
main()
'''
