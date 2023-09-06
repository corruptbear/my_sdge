#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import csv
import numpy as np
import pandas as pd
import datetime

# for holiday exclusion
from pandas.tseries.holiday import USFederalHolidayCalendar

# for 3d bar plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from itertools import chain

# for num2date
import matplotlib.dates as mpl_dates

# for FuncFormatter
import matplotlib.ticker as ticker


def get_baseline(zone=None, season=None, service_type=None, multiplier=1, billing_days=30):
    # source: https://www.sdge.com/baseline-allowance-calculator
    zone_index_mapping = {"coastal": 0, "inland": 1, "mountain": 2, "desert": 3}
    zone_index = zone_index_mapping[zone]

    summer_electric = [6, 8.7, 15.2, 17]
    winter_electric = [8.8, 12.2, 22.1, 17.1]

    summer_combined = [9.0, 10.4, 13.6, 15.9]
    winter_combined = [9.2, 9.6, 12.9, 10.9]

    daily_baseline = {
        "electric": {
            "summer": summer_electric,
            "winter": winter_electric,
        },
        "combined": {
            "summer": summer_combined,
            "winter": winter_combined,
        },
    }
    return int(np.floor(multiplier * billing_days * daily_baseline[service_type][season][zone_index]))


# https://www.sdge.com/regulatory-filing/16026/residential-time-use-periods
def get_tou1_hours(date):
    is_march_or_april = 1 if (date.month == 3 or date.month == 4) else 0

    # non-holiday weekdays
    WEEKDAY = {"SUPER_OFFPEAK": [0, 1, 2, 3, 4, 5], "OFFPEAK": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}
    # weekends and holidays
    HOLIDAY = {"SUPER_OFFPEAK": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "OFFPEAK": [14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}

    if is_march_or_april:
        WEEKDAY["SUPER_OFFPEAK"] += [10, 11, 12, 13]
        WEEKDAY["OFFPEAK"] = [6, 7, 8, 9, 14, 15, 21, 22, 23]

    # which day is it?
    weekday = date.weekday()

    # mark US holidays
    cal = USFederalHolidayCalendar()
    start = datetime.datetime(date.year, 1, 1)
    end = datetime.datetime(date.year + 1, 1, 1)
    holidays = cal.holidays(start=start, end=end).to_pydatetime()

    if weekday == 5 or weekday == 6 or date in holidays:
        return HOLIDAY
    else:
        return WEEKDAY


def get_tou2_hours(date):
    EVERYDAY = {"OFFPEAK": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}
    return EVERYDAY


def get_non_tou_hours(date):
    EVERYDAY = {"FLAT": [i for i in range(24)]}
    return EVERYDAY


def extract_dates(daily):
    return [pd.to_datetime(x[0], "%Y-%m-%d").date() for x in daily.items()]


def tou_stacked_plot(daily=None):
    dates = extract_dates(daily)
    daily_arrays = {l: np.array([]) for l in get_tou_hours.labels}

    for date, consumption_data in daily.items():
        d = pd.to_datetime(date, "%Y-%m-%d").date()

        for category in daily_arrays:
            current_array = daily_arrays[category]
            daily_arrays[category] = np.append(current_array, sum([consumption_data[hour] for hour in get_tou_hours(d)[category]]))

    # plot the daily summary with stacked bars
    # plt.figure()

    previous = np.zeros(len(dates))
    for index, category in enumerate(daily_arrays):
        print(index, category, daily_arrays[category])
        plt.bar(dates, daily_arrays[category], label=category, color=f"C{index+1}", bottom=previous)
        previous += daily_arrays[category]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.ylabel("Consumption (kWh)")
    plt.grid(linestyle="--", axis="y")
    plt.title("Daily Consumption")
    plt.legend()
    plt.tight_layout()


def aggregated_hourly_plot(daily=None):
    dates = extract_dates(daily)
    # plot the hourly summary
    plt.figure()
    plt.title(f'Aggregated Hourly Consumption: {dates[0].strftime("%Y/%m/%d")} to {dates[-1].strftime("%Y/%m/%d")}')

    hourly = [sum([daily[x][i] for x in daily.index]) for i in range(24)]

    plt.bar(list(range(24)), hourly)
    plt.ylabel("Consumption (kWh)")
    plt.xlim([-0.5, 23.5])
    plt.show()


def daily_hourly_2d_plot(daily=None):
    if len(daily.index) >= 50:
        return
    dates = extract_dates(daily)
    fig, axs = plt.subplots(len(daily.index), 1, sharex=True)

    i = 0
    # series can use iteritems method
    for date, consumption_data in daily.items():
        # daw plot for a particular day
        axs[i].bar(list(range(24)), consumption_data)
        axs[i].set_yticks([])
        i += 1
        # collect flattened data for 3d plot

    plt.xlim([-0.5, 23.5])

    """
    #add the common Y label before plt 3.4.0
    fig.add_subplot(111, frameon=False)
    #hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel("consumption by day")
    """
    # add the common Y label after matplotlib 3.4.0
    fig.supylabel("Consumption by Day")
    fig.suptitle(f'Daily Details 2D: {dates[0].strftime("%Y/%m/%d")} to {dates[-1].strftime("%Y/%m/%d")}')


def daily_hourly_3d_plot(daily=None):
    if len(daily.index) >= 50:
        return
    dates = extract_dates(daily)
    all_data = list(chain.from_iterable(daily))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 24 hours
    xvalues = np.array(list(range(24)))
    # the days, the trick here is to convert dates to number, for easier 3d plot involving dates
    yvalues = np.array([mpl_dates.date2num(d) for d in daily.index])
    # yvalues = np.array(list(range(len(daily.index))));
    xx, yy = np.meshgrid(xvalues, yvalues)

    xx = xx.flatten()
    yy = yy.flatten()

    # convert to np array
    all_data = np.array(all_data)
    zz = np.zeros_like(all_data)

    dx = np.ones_like(xx)
    dy = np.ones_like(yy)
    dz = all_data

    # colorcode the bars
    colors = plt.cm.jet(all_data / float(all_data.max()))

    ax.set_xlim([-0.5, 23.5])
    ax.set_ylim([min(yvalues), max(yvalues)])

    ax.set_xlabel("Hour")
    ax.set_zlabel("Consumption (kWh)")

    # ylabels=[a.strftime('%Y-%m-%d') for a in days ]

    ax.bar3d(xx, yy, zz, dx, dy, dz, color=colors)

    # The function should take in two inputs (a tick value x and a position pos), and return a string containing the corresponding tick label.
    num2formatted = lambda x, _: mpl_dates.num2date(x).strftime("%Y-%m-%d")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(num2formatted))

    # auto-adjust the orientation of labels
    # fig.autofmt_xdate()
    # manually set the orientation of labels
    ax.tick_params(axis="y", labelrotation=90)
    plt.title(f'Daily Details 3D: {dates[0].strftime("%Y/%m/%d")} to {dates[-1].strftime("%Y/%m/%d")}')


def plot_sdge_hourly(filename):
    # read the csv and skip the first rows
    df = pd.read_csv(
        filename,
        skiprows=13,
        index_col=False,
        usecols=["Date", "Start Time", "Consumption"],
        skipinitialspace=True,
        dtype={"Consumption": np.float32},
        parse_dates=["Date"],
    )

    # group the hourly data by dates, get a pandas series, which is 1-dimensional with label
    # daily_summary = df.groupby('Date')['Consumption'].sum()

    daily = df.groupby("Date")["Consumption"].apply(list)

    # plot the daily summary bar plot without stacking
    # plt.bar(days,daily_summary.values)

    tou_stacked_plot(daily=daily)

    # plot day by day
    daily_hourly_2d_plot(daily=daily)
    daily_hourly_3d_plot(daily=daily)

    # plot hourly data summed across days
    aggregated_hourly_plot(daily=daily)


if __name__ == "__main__":
    plan_mapping = {"TOU1": get_tou1_hours, "TOU2": get_tou2_hours, "FLAT": get_non_tou_hours}
    get_tou1_hours.labels = ["SUPER_OFFPEAK", "OFFPEAK", "PEAK"]
    get_tou2_hours.labels = ["OFFPEAK", "PEAK"]
    get_non_tou_hours.labels = ["FLAT"]

    PLAN = "TOU1"
    get_tou_hours = plan_mapping[PLAN]
    print(get_tou_hours.labels)

    print(get_baseline(zone="desert", season="winter", service_type="combined", multiplier=1.3))
    filename = "/Users/lws/Downloads/sdge_data/Electric_60_Minute_2-1-2023_2-28-2023_20230819.csv"
    plot_sdge_hourly(filename)
