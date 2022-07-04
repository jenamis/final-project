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
