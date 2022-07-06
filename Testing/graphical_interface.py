from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import ttk
from time import sleep
from tkinter import *
from tkinter.ttk import *
import time


def gui_input_file():
    tk.messagebox.showinfo(title="Select SMILES List (CSV)", message="Select the list of structures' SMILES to process. NOTE: Column header must be 'SMILES'.")
    inputfile = fd.askopenfilename()
    return inputfile

def gui_outputfile_file():
    outputfile = asksaveasfile()
    return outputfile

def gui_loading_start():
    ws = Tk()
    ws.title('PythonGuides')
    ws.geometry('400x250+1000+300')


    def step():
        for i in range(5):
            ws.update_idletasks()
            pb1['value'] += 20

            time.sleep(1)


    pb1 = Progressbar(ws, orient=HORIZONTAL, length=100, mode='indeterminate')
    pb1.pack(expand=True)

    Button(ws, text='Start', command=step).pack()

    ws.mainloop()

    teams = range(100)

def progress_bar():

    def button_command():
        #start progress bar
        popup = tk.Toplevel()
        tk.Label(popup, text="Files being downloaded").grid(row=0,column=0)

        progress = 0
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()
        teams = range(100)
        progress_step = float(100.0/len(teams))
        for team in teams:
            popup.update()
            sleep(5) # lauch task
            progress += progress_step
            progress_var.set(progress)

        return 0

    root = tk.Tk()

    tk.Button(root, text="Launch", command=button_command).pack()

    root.mainloop()