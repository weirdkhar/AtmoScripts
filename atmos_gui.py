'''
Code for the gui of the processing of aerosol gear

Written by Ruhi Humphries
2017

Useful documentation: 
    http://www.tkdocs.com/tutorial/widgets.html 
    http://pyinmyeye.blogspot.com.au/2012/08/tkinter-combobox-demo.html
'''
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
global files_raw, default_output_path

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