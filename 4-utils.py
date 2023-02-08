
#Import Modules

#Imports the pandas module and refers it as pd
import pandas as pd
#Imports the submodule datetime from the module datetime
from datetime import datetime
#Imports the sys module
import sys
#Imports the tkinter module, used for GUI
from tkinter import *
#Import all the functions defined in the utils.py file
from utils import *
#Import the numpy module as np
import numpy as np
#Import the linear_model submodule from sklearn
from sklearn import linear_model
#Import the pyplot submodule from matplotlib as plt
from matplotlib import pyplot as plt

#--------------------------------
#This function gets the MD dataset and changes it into usable format
def MDgetdata():

    #Defines the filename of the file ('Mdgov.csv') that is to be loaded
    fileName = 'MDgov.csv'

    #Reads the csv data file into a dataframe called vaccData, which is short for Vaccine Data
    vaccData = pd.read_csv(fileName)

    #Removes all the columns besides the Day and CompletedVaxCumulative
    vaccData=vaccData.drop(columns=['OBJECTID','FirstDailyDose','FirstDoseCumulative','SecondDailyDose','SecondDoseCumulative','SingleDailyDose', 'SingleDoseCumulative', 'AtleastOneVaccine', 'AtleastOneVaccineCumulative', 'CompletedVax', 'FirstSecondSingleVaccinations', 'FirstSecondSingleVaccinationsCu'])

    #Only extracts rows that correspond to the USA
    #vaccData = vaccData[vaccData['CountryID']=='USA']

    #Uses For loop  to iterate each row using vaccData dataframe
    #Uses the datetime module to convert dates (which are strings) into something that is convenient and can be used for curve fitting, like integers or floats.
    for iRow in vaccData.index:
        #Extracts the date corresponding to iRow
        currentDateStr=vaccData.at[iRow,'Day']

        #Skips the months December, January and February by substringing a string and using a continue statement
        if currentDateStr[5:7] == "12":
          continue
        if currentDateStr[5:7] == "01":
          continue
        if currentDateStr[5:7] == "02":
          continue
        
        #vaccData['Day'] = vaccData['Day'].str.replace(r'15:00:00+00', '')

        #Uses the strptime method to convert the date string into a datetime object
        currentDate=datetime.strptime(currentDateStr,'%Y/%m/%d %H:%M:%S+%f')

        # Define dayZero. Even though the data starts in 2020, I want to use a smaller portion of the data to fit the line better to it. So I'm going to subtract off the date February 1, 2021 from all the dates. This means that all dates will be relative to February 1 of this year.
        dayZero=datetime(2021,3,1)

        # Take the difference between the current date and the first day of data recorded
        correctedDate=currentDate-dayZero

        # Since this particular data set is reporting no more than once per day, I will use days as my units for time. Note, this does not mean that there is an entry for every day necessarily.
        correctedDate.days

        # Add this corrected date which is an integer representing the number of days since February 1, 2021 and add it to a new column in the dataframe
        vaccData.at[iRow,'correctedDay']=correctedDate.days
        
    #Divides the data by 2 so that it represents the number of people fully vaccinated in millions because each person gets 2 doses
    #vaccData['doses_administered'] = vaccData['doses_administered']/2

    #Divides the data by 1 million so that it is easier to read without changing the data (Ex. A number such as 21,368,790 would turn into 21.368790)
    vaccData['CompletedVaxCumulative'] = vaccData['CompletedVaxCumulative']/1e6

    #Drops all rows with NaN values in the 'correctedDay' column by modifiying the original dataframe (vaccData)
    vaccData.dropna(inplace=True, subset=['correctedDay'])
    
    #Returns the augmented dataframe    
    return vaccData

#-----------------------------------

#This function loads, extracts, and graphs the MD dataset to estimate Herd Immunity 
def MDgraphData():

  #-----------------------------------
  # Load Data

  # See the utils file for information on getdata()
  # Stores the dataframe from the function getdata into the variable vaccData
  vaccData=MDgetdata()

  # Prints the dataframe in the console
  print(vaccData)

  #-----------------------------------
  # Extract Data to Fit

  '''Reshape the array to be the shape that LinearRegression().fit() expects later on. See the documentation on LinearRegression().fit() to see what it says and see the documentation on reshape (https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape) to understand what the two arguments mean.'''
  # Extract the correctedDay as the x data and reshapes the data into 1 column no matter how many datapoints there are
  # Convert to a numpy array
  x=vaccData['correctedDay'].to_numpy().reshape(-1,1)

  # Extract the doses_administered as the y date. 
  # Convert to a numpy array
  y=vaccData['CompletedVaxCumulative'].to_numpy()

  #-----------------------------------
  # Create Linear Regression Object

  #https://scikit-learn.org/stable/modules/linear_model.html
  #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

  model = linear_model.LinearRegression()

  # Use the fit method to find the parameters that fit a line to the data. Notice that nothing is returned because model is an object and the fit method modifies the attributes of that object (recall the lesson on object oriented programming)
  model.fit(x, y)

  # Extract the slope from the fitted model
  m=model.coef_

  # Extract the y-intercept from the fitted model
  b=model.intercept_

  #-----------------------------------
  # Estimate the Day When Herd Immunity will be Reached

  # Set the threshold percentage for herd immunity
  # The "star" label is usually used to denote special values
  #70% percent of the Maryland population
  yStar=4.231976

  # Use a function that will take m, b, and yStar and solve for the xStar
  # See the utils file for the definition of solvelinex
  xStar=solvelinex(float(m),float(b),yStar)

  #-----------------------------------
  # Plot Results
  # Plot the raw data

  # Scatter plot x and y using black and label the data 
  plt.scatter(x, y,  color='black', label='Raw Data')

  #-----------------------------------
  # Plot a line of best fit

  # Get the first and last values of x
  xFit=x[[0,-1]]

  # Use the equation of the line to find the corresponding y values
  # Note that xFit is an array. numpy allow you to do matrix operations so that you don't have to do this with a loop.
  yFit = m * xFit + b

  # Plot the line using two points, color it blue with a line width of 3 and label it
  plt.plot(xFit,yFit, color='blue', linewidth=3, label='Fitted Data')

  #-----------------------------------
  # Plot the extrapolated line that goes up the herd immunity threshold

  # since x and xFit are numpy arrays, using the append method to create a new numpy array
  xEx=np.append(xFit[-1], xStar)
  yEx=np.append(yFit[-1], yStar)

  # Plot the line
  plt.plot(xEx,yEx, color='red', linewidth=3, label='Extrapolated Data')

  #-----------------------------------
  # Plot a threshold line

  xT=np.append(x[0], xStar)
  yT=[yStar, yStar]

  # Plot the line
  plt.plot(xT,yT, color='green', linewidth=3, label='Herd Immunity Threshold')

  # Add a title
  plt.title('Vaccination Trend in Maryland')

  # Add an xlabel
  plt.xlabel('Days since 1 March 2021')

  # Add a y label
  plt.ylabel('People Fully Vaccinated (Millions)')

  # Add a legend (legend knows what to add because of the label='string' in each plot call)
  plt.legend()

  #-----------------------------------
  # Print out the Results

  print(f'Maryland is expected to reach herd immunity on day {int(xStar + 60)} of 2021.')

  #Adjust plot parameters so they fit in the visualization
  plt.tight_layout()

  # Show the plot (note this is a blocking function and causes the code to pause here. Click the X in the figure to continue.)
  plt.show()

#-----------------------------------

#This function gets the NY dataset and changes it into usable format
def NYgetdata():

    #Defines the filename of the file ('NYgit.csv') that is to be loaded
    fileName = 'NYgit.csv'

    #Reads the csv data file into a dataframe called vaccData, which is short for Vaccine Data
    vaccData = pd.read_csv(fileName)

    #Removes all the columns besides the Day and doses_administered
    vaccData=vaccData.drop(columns=['ADMIN_DOSE1_DAILY','ADMIN_DOSE1_CUMULATIVE','ADMIN_DOSE2_DAILY','ADMIN_DOSE2_CUMULATIVE','ADMIN_SINGLE_DAILY','ADMIN_SINGLE_CUMULATIVE', 'ADMIN_ALLDOSES_DAILY', 'ADMIN_ALLDOSES_7DAYAVG', 'INCOMPLETE'])

    #Only extracts rows that correspond to the USA
    #vaccData = vaccData[vaccData['CountryID']=='USA']

    #Uses For loop  to iterate each row using vaccData dataframe
    #Uses the datetime module to convert dates (which are strings) into something that is convenient and can be used for curve fitting, like integers or floats.
    for iRow in vaccData.index:
        #Extracts the date corresponding to iRow
        currentDateStr=vaccData.at[iRow,'Day']

        #Skips the months December, January and February by substringing a string and using a continue statement
        if currentDateStr[5:7] == "12":
          continue
        if currentDateStr[5:7] == "01":
          continue
        if currentDateStr[5:7] == "02":
          continue
        
        #vaccData['Day'] = vaccData['Day'].str.replace(r'15:00:00+00', '')

        #Uses the strptime method to convert the date string into a datetime object
        currentDate=datetime.strptime(currentDateStr,'%Y-%m-%d')

        # Define dayZero. Even though the data starts in 2020, I want to use a smaller portion of the data to fit the line better to it. So I'm going to subtract off the date February 1, 2021 from all the dates. This means that all dates will be relative to February 1 of this year.
        dayZero=datetime(2021,3,1)

        # Take the difference between the current date and the first day of data recorded
        correctedDate=currentDate-dayZero

        # Since this particular data set is reporting no more than once per day, I will use days as my units for time. Note, this does not mean that there is an entry for every day necessarily.
        correctedDate.days

        # Add this corrected date which is an integer representing the number of days since February 1, 2021 and add it to a new column in the dataframe
        vaccData.at[iRow,'correctedDay']=correctedDate.days
        
    #Divides the data by 2 so that it represents the number of people fully vaccinated in millions because each person gets 2 doses
    vaccData['doses_administered'] = vaccData['doses_administered']/2

    #Divides the data by 1 million so that it is easier to read without changing the data (Ex. A number such as 21,368,790 would turn into 21.368790)
    vaccData['doses_administered'] = vaccData['doses_administered']/1e6

    #Drops all rows with NaN values in the 'correctedDay' column by modifiying the original dataframe (vaccData)
    vaccData.dropna(inplace=True, subset=['correctedDay'])
    
    #Returns the augmented dataframe    
    return vaccData

#-----------------------------------

#This function loads, extracts, and graphs the NY dataset to estimate Herd Immunity 
def NYgraphData():

  #-----------------------------------
  # Load Data

  # See the utils file for information on getdata()
  # Stores the dataframe from the function getdata into the variable vaccData
  vaccData=NYgetdata()

  # Prints the dataframe in the console
  print(vaccData)

  #-----------------------------------
  # Extract Data to Fit

  '''Reshape the array to be the shape that LinearRegression().fit() expects later on. See the documentation on LinearRegression().fit() to see what it says and see the documentation on reshape (https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape) to understand what the two arguments mean.'''
  # Extract the correctedDay as the x data and reshapes the data into 1 column no matter how many datapoints there are
  # Convert to a numpy array
  x=vaccData['correctedDay'].to_numpy().reshape(-1,1)

  # Extract the doses_administered as the y date. 
  # Convert to a numpy array
  y=vaccData['doses_administered'].to_numpy()

  #-----------------------------------
  # Create Linear Regression Object

  #https://scikit-learn.org/stable/modules/linear_model.html
  #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

  model = linear_model.LinearRegression()

  # Use the fit method to find the parameters that fit a line to the data. Notice that nothing is returned because model is an object and the fit method modifies the attributes of that object (recall the lesson on object oriented programming)
  model.fit(x, y)

  # Extract the slope from the fitted model
  m=model.coef_

  # Extract the y-intercept from the fitted model
  b=model.intercept_

  #-----------------------------------
  # Estimate the Day When Herd Immunity will be Reached

  # Set the threshold percentage for herd immunity
  # The "star" label is usually used to denote special values
  #70% percent of the New York population
  yStar=5.835772

  # Use a function that will take m, b, and yStar and solve for the xStar
  # See the utils file for the definition of solvelinex
  xStar=solvelinex(float(m),float(b),yStar)

  #-----------------------------------
  # Plot Results
  # Plot the raw data

  # Scatter plot x and y using black and label the data 
  plt.scatter(x, y,  color='black', label='Raw Data')

  #-----------------------------------
  # Plot a line of best fit

  # Get the first and last values of x
  xFit=x[[0,-1]]

  # Use the equation of the line to find the corresponding y values
  # Note that xFit is an array. numpy allow you to do matrix operations so that you don't have to do this with a loop.
  yFit = m * xFit + b

  # Plot the line using two points, color it blue with a line width of 3 and label it
  plt.plot(xFit,yFit, color='blue', linewidth=3, label='Fitted Data')

  #-----------------------------------
  # Plot the extrapolated line that goes up the herd immunity threshold

  # since x and xFit are numpy arrays, using the append method to create a new numpy array
  xEx=np.append(xFit[-1], xStar)
  yEx=np.append(yFit[-1], yStar)

  # Plot the line
  plt.plot(xEx,yEx, color='red', linewidth=3, label='Extrapolated Data')

  #-----------------------------------
  # Plot a threshold line

  xT=np.append(x[0], xStar)
  yT=[yStar, yStar]

  # Plot the line
  plt.plot(xT,yT, color='green', linewidth=3, label='Herd Immunity Threshold')

  # Add a title
  plt.title('Vaccination Trend in New York City')

  # Add an xlabel
  plt.xlabel('Days since 1 March 2021')

  # Add a y label
  plt.ylabel('People Fully Vaccinated (Millions)')

  # Add a legend (legend knows what to add because of the label='string' in each plot call)
  plt.legend()

  #-----------------------------------
  # Print out the Results

  print(f'New York City is expected to reach herd immunity on day {int(xStar + 60)} of 2021.')

  #Adjust plot parameters so they fit in the visualization
  plt.tight_layout()

  # Show the plot (note this is a blocking function and causes the code to pause here. Click the X in the figure to continue.)
  plt.show()

#-----------------------------------

#Solvelinex is a function that will take the slope (m), the y-intercept (b), and a value y and solve for the corresponding x value (gets the x vakue when herd immunity is achieved) 
def solvelinex(m,b,y):
    #Check that m is an integer or a float and is not equal to zero
    #Makes sure that the slope of the line isn't 0 because that would show no progress/x-axis
    check_m = (type(m)==int or type(m)==float) and m!=0
    # Similar
    check_b = type(b)==int or type(b)==float
    # similar
    check_y = type(y)==int or type(y)==float

    # If all the checks were True then calculate the output
    if check_m and check_b and check_y:
        return (y-b)/m
    # Else raise an error
    else:
        raise ValueError('m, b, or x are not the proper type or m is zero')




