
#Import Modules

#Import all the functions defined in the utils.py file
from utils import *
#Import the pandas module as pd
import pandas as pd
#Import the numpy module as np
import numpy as np
#Import the linear_model submodule from sklearn
from sklearn import linear_model
#Import the pyplot submodule from matplotlib as plt
from matplotlib import pyplot as plt
#Import the sys submodule from os
from os import sys
##Imports the tkinter mosule, used for GUI
from tkinter import *

#--------------------------------
#Creating the GUI

#Creates the root widget
root = Tk()

#Gives a function to the button
#Destroys all of the widgets and continues the program
def Break():
  root.destroy()

optionLabel = Label(root, text="Which area do you want data from?")

#Creates a button that proceeds to the program when clicked on
userButton = Button(root, text="Maryland", activeforeground="blue", bd=3, padx=25, pady=20, command=MDgraphData)

userButton2 = Button(root, text="New York City", activeforeground="blue", bd=3, padx=7, pady=20, command=NYgraphData)

userButton3 = Button(root, text="Exit", activeforeground="red", bd=3, padx=15, pady=7, command=Break)

#Puts the buttons and label on the screen
optionLabel.pack()
userButton.pack()
userButton2.pack()
userButton3.pack()
root.mainloop()

#--------------------------------
