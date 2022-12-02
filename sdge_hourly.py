#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import csv
import numpy as np
import pandas as pd
import datetime
#for holiday exclusion
from pandas.tseries.holiday import USFederalHolidayCalendar
#for 3d bar plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
#for num2date 
import matplotlib.dates as dates
#for FuncFormatter
import matplotlib.ticker as ticker

#https://www.sdge.com/regulatory-filing/16026/residential-time-use-periods
def get_tou_hours(date):
    is_march_or_april = 1 if (date.month == 3 or date.month == 4) else 0
    
    #non-holiday weekdays
    WEEKDAY={'SUPER_OFFPEAK':[0,1,2,3,4,5], 'OFFPEAK':[6,7,8,9,10,11,12,13,14,15,21,22,23],'PEAK':[16,17,18,19,20]}
    #weekends and holidays
    OTHER={'SUPER_OFFPEAK':[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 'OFFPEAK':[14,15,21,22,23], 'PEAK':[16,17,18,19,20]}

    
    if is_march_or_april:
        WEEKDAY['SUPER_OFFPEAK']+=[10,11,12,13]
        WEEKDAY['OFFPEAK']=[6,7,8,9,14,15,21,22,23]     
        
    #which day is it?    
    weekday = date.weekday()
    
    #mark US holidays
    cal = USFederalHolidayCalendar()
    start = datetime.datetime(date.year,1,1)
    end = datetime.datetime(date.year + 1,1,1)
    holidays = cal.holidays(start=start, end=end).to_pydatetime()
    
    if weekday == 5 or weekday == 6 or date in holidays:
        return OTHER 
    else:
        return WEEKDAY
            

def plot_sdge_hourly(filename):
    #read the csv and skip the first rows
    df = pd.read_csv(filename, skiprows=13,index_col=False, usecols=['Date','Start Time','Consumption'],skipinitialspace=True, dtype={'Consumption':np.float32}, parse_dates=['Date'])

    #group the hourly data by dates, get a pandas series, which is 1-dimensional with label
    #daily_summary = df.groupby('Date')['Consumption'].sum()
    
    daily = df.groupby('Date')['Consumption'].apply(list)

    #days = [pd.to_datetime(a,'%Y-%m-%d').date() for a in daily_summary.index.tolist()]
    
    days = []
    daily_super_offpeak = []
    daily_offpeak = []
    daily_peak = []
    
    for date, consumption_data in daily.iteritems():
        d = pd.to_datetime(date,'%Y-%m-%d').date()
            
        days.append(d)
        daily_super_offpeak.append(sum([consumption_data[i] for i in get_tou_hours(d)['SUPER_OFFPEAK']]))
        daily_offpeak.append(sum([consumption_data[i] for i in get_tou_hours(d)['OFFPEAK']]))
        daily_peak.append(sum([consumption_data[i] for i in get_tou_hours(d)['PEAK']]))    
    
    #convert to np so that they could be stacked for bar plot
    daily_super_offpeak = np.array(daily_super_offpeak)
    daily_offpeak = np.array(daily_offpeak)
    daily_peak = np.array(daily_peak)
    
    #plot the daily summary bar plot without stacking
    #plt.bar(days,daily_summary.values)
    
    #plot the daily summary with stacked bars
    plt.figure()
    
    #C0, C1, C2 indicate the default colors in the color cycles
    plt.bar(days, daily_super_offpeak, label='super offpeak', color = 'C0')
    plt.bar(days, daily_offpeak, bottom = daily_super_offpeak, label='offpeak', color = 'C2')
    plt.bar(days, daily_peak, bottom = daily_super_offpeak + daily_offpeak, label='peak', color = 'C1')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.ylabel('Consumption (kWh)')
    plt.grid(linestyle='--',axis='y')
    plt.title('Daily Consumption')
    plt.legend()
    plt.tight_layout()

    #plot day by day
    
    #omit this plot when there are too many days
    if (len(daily.index)<50):
        fig, axs = plt.subplots(len(daily.index), 1, sharex=True)
        
        all_data = []
    
        i = 0
        #series can use iteritems method
        for date, consumption_data in daily.iteritems():
            #daw plot for a particular day
            axs[i].bar(list(range(24)),consumption_data)
            axs[i].set_yticks([])
            i+=1
            #collect flattened data for 3d plot
            all_data = all_data + consumption_data 

        plt.xlim([-0.5, 23.5])    

        """
        #add the common Y label before plt 3.4.0
        fig.add_subplot(111, frameon=False)
        #hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.ylabel("consumption by day")
        """
        #add the common Y label after matplotlib 3.4.0
        fig.supylabel('Consumption by Day')
        fig.suptitle(f'Daily Details 2D: {days[0].strftime("%Y/%m/%d")} to {days[-1].strftime("%Y/%m/%d")}')
               
        #3d plot?    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #24 hours
        xvalues = np.array(list(range(24)));
        #the days, the trick here is to convert dates to number, for easier 3d plot involving dates
        yvalues= np.array([dates.date2num(d) for d in daily.index])
        #yvalues = np.array(list(range(len(daily.index))));
        xx, yy = np.meshgrid(xvalues, yvalues)

        xx = xx.flatten()
        yy = yy.flatten()
        
        #convert to np array
        all_data = np.array(all_data)
        zz = np.zeros_like(all_data)
               
        dx = np.ones_like(xx)
        dy = np.ones_like(yy)
        dz = all_data
        
        #colorcode the bars
        colors = plt.cm.jet(all_data /float(all_data.max()))
               
        ax.set_xlim([-0.5,23.5])
        ax.set_ylim([min(yvalues),max(yvalues)])
                
        ax.set_xlabel('Hour')
        ax.set_zlabel('Consumption (kWh)')
     
        #ylabels=[a.strftime('%Y-%m-%d') for a in days ]
        
        ax.bar3d(xx,yy,zz,dx,dy,dz, color = colors)
        
        #The function should take in two inputs (a tick value x and a position pos), and return a string containing the corresponding tick label.
        num2formatted =  lambda x, _: dates.num2date(x).strftime('%Y-%m-%d')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(num2formatted))
        
        #auto-adjust the orientation of labels
        #fig.autofmt_xdate()    
        #manually set the orientation of labels 
        ax.tick_params(axis='y', labelrotation = 90)
        plt.title(f'Daily Details 3D: {days[0].strftime("%Y/%m/%d")} to {days[-1].strftime("%Y/%m/%d")}')
        
    #plot the hourly summary
    plt.figure()
    plt.title(f'Aggregated Hourly Consumption: {days[0].strftime("%Y/%m/%d")} to {days[-1].strftime("%Y/%m/%d")}')
    
    hourly=[sum([daily[x][i] for x in daily.index]) for i in range(24)]
    
    plt.bar(list(range(24)),hourly)
    plt.ylabel('Consumption (kWh)')
    plt.xlim([-0.5, 23.5])  
    plt.show()


if __name__ == "__main__":
    filename="/Users/lws/Downloads/Electric_60_Minute_9-1-2022_10-7-2022_20221028.csv"
    plot_sdge_hourly(filename)