#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime
import yaml
import traceback
import os
from functools import cache
from collections import namedtuple
import click

# for holiday exclusion
from pandas.tseries.holiday import USFederalHolidayCalendar

# for 3d bar plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from itertools import chain

# for num2date
import matplotlib.dates as mpl_dates

# for FuncFormatter
import matplotlib.ticker as ticker


def load_yaml(filepath):
    """
    Load the yaml file. Returns an empty dictionary if the file cannot be read.
    """
    # yaml_path = os.path.join(pwd, filepath)
    try:
        with open(filepath, "r") as stream:
            dictionary = yaml.safe_load(stream)
            return dictionary
    except:
        traceback.print_exc()
        return dict()


SDGEDay = namedtuple("SDGEDate", ["date", "season"])

pwd = os.path.dirname(os.path.realpath(__file__))
rates_path = os.path.join(pwd, "sdge_rates_2023.yaml")
rates = load_yaml(rates_path)


class SDGECaltulator:
    def __init__(self, daily_24h, plan="TOU-DR1", zone="coastal", service_type="electric", pcia_rate=0.01687, billing_cycles=1):
        self.daily_24h = daily_24h
        self.days = [SDGEDay(date, get_season(date)) for date in extract_dates(self.daily_24h)]
        self.zone = zone
        self.pcia_rate = pcia_rate
        self.billing_cycles = billing_cycles
        self.service_type = service_type
        self.total_usage = sum([sum(usage) for date, usage in self.daily_24h.items()])

        print(f"starting:{self.days[0].date} ending:{self.days[-1].date}")
        print(f"{len(self.days)} days, {len([x for x in self.days if x.season=='summer'])} summer days, {len([x for x in self.days if x.season=='winter'])} winter days")
        print(f"total_usage:{self.total_usage:.4f} kWh")
        print(f"number of billing cycles:{self.billing_cycles}")

        assert self.days[0].date.year == self.days[-1].date.year, "all data must be from the same year"

    @cache
    def tally(self, schedule=None):
        daily_arrays = category_tally_by_schedule(daily=self.daily_24h, schedule=schedule)
        rates_classes = schedule.rates_classes

        season_days_counter = {"summer": 0, "winter": 0}
        # tally the summer usage and winter usage
        season_class_tally = {"summer": {x: 0.0 for x in rates_classes}, "winter": {x: 0.0 for x in rates_classes}}
        for k, day in enumerate(self.days):
            season_days_counter[day.season] += 1
            for rate_class in rates_classes:
                season_class_tally[day.season][rate_class] += daily_arrays[rate_class][k]
        return rates_classes, season_days_counter, season_class_tally

    def calculate(self, plan=None):
        # usage tally
        rates_classes, season_days_counter, season_class_tally = self.tally(schedule=rates_schedules[plan])
        #print(season_class_tally)

        total_fee = 0.0

        for season in ["winter", "summer"]:
            season_total_usage = sum(season_class_tally[season].values())

            raw = get_raw_sum(season_class_tally[season], rates[plan][season])
            total_fee += raw

            allowance_deduction = get_allowance_deduction(
                zone=self.zone,
                season=season,
                service_type=self.service_type,
                billing_days=season_days_counter[season],
                total_usage=season_total_usage,
                credit_per_kwh=rates[plan]["credit"],
            )
            # remove the deduction
            total_fee -= allowance_deduction
        # apply the recurring service fee
        total_fee += rates[plan]["service_fee"] * self.billing_cycles
        # apply the PCIA rates for CCA
        if "CCA" in plan:
            total_fee += self.total_usage * self.pcia_rate
        return total_fee


def calculate_misc_fees(total_usage=0.0, pcia_rate=0.01687):
    misc_fee = 0.0

    return misc_fee


def get_raw_sum(usage_by_class, rates_by_class):
    """
    usage_by_class (dict)
    rates_by_class (dict)
    """
    return sum([usage_by_class[rates_class] * rates_by_class[rates_class] for rates_class in usage_by_class])


@cache
def get_allowance_deduction(zone="coastal", season=None, service_type="electric", billing_days=30, total_usage=0.0, credit_per_kwh=0.11724):
    # calculate 130% allowance deduction
    baseline130 = get_baseline(zone=zone, season=season, service_type=service_type, multiplier=1.3, billing_days=billing_days)
    deducted_usage = min(total_usage, baseline130)
    # calculate deduction
    allowance_deduction = credit_per_kwh * deducted_usage
    return allowance_deduction


@cache
def get_baseline(zone=None, season=None, service_type="electric", multiplier=1.3, billing_days=30):
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


def get_season(date):
    if date.month in {6, 7, 8, 9, 10}:
        return "summer"
    return "winter"


# https://www.sdge.com/regulatory-filing/16026/residential-time-use-periods
@cache
def schedule_sop(date):
    """
    rates schedule for plans with SUPER OFFPEAK, OFFPEAK, PEAK rates
    """
    is_march_or_april = 1 if (date.month == 3 or date.month == 4) else 0

    # non-holiday weekdays
    WEEKDAY_HOURS = {"SUPER_OFFPEAK": [0, 1, 2, 3, 4, 5], "OFFPEAK": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}
    # weekends and holidays
    HOLIDAY_HOURS = {"SUPER_OFFPEAK": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "OFFPEAK": [14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}

    if is_march_or_april:
        WEEKDAY_HOURS["SUPER_OFFPEAK"] += [10, 11, 12, 13]
        WEEKDAY_HOURS["OFFPEAK"] = [6, 7, 8, 9, 14, 15, 21, 22, 23]

    # which day is it?
    weekday = date.weekday()

    # mark US holidays
    cal = USFederalHolidayCalendar()
    start = datetime.datetime(date.year, 1, 1)
    end = datetime.datetime(date.year + 1, 1, 1)
    holidays = cal.holidays(start=start, end=end).to_pydatetime()

    if weekday == 5 or weekday == 6 or date in holidays:
        return HOLIDAY_HOURS
    return WEEKDAY_HOURS


@cache
def schedule_op(date):
    """
    rates schedule for plans with OFFPEAK, PEAK rates
    """
    EVERYDAY_HOURS = {"OFFPEAK": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23], "PEAK": [16, 17, 18, 19, 20]}
    return EVERYDAY_HOURS


@cache
def schedule_flat(date):
    """
    rates schedule for non-TOU plans
    """
    EVERYDAY_HOURS = {"FLAT": [i for i in range(24)]}
    return EVERYDAY_HOURS


rates_schedules = {
    "TOU-DR1": schedule_sop,
    "TOU-DR2": schedule_op,
    "EV-TOU-5": schedule_sop,
    "EV-TOU-2": schedule_sop,
    "DR": schedule_flat,
    "CCA-TOU-DR1": schedule_sop,
    "CCA-TOU-DR2": schedule_op,
    "CCA-EV-TOU-5": schedule_sop,
    "CCA-EV-TOU-2": schedule_sop,
    "CCA-DR": schedule_flat,
}
schedule_sop.rates_classes = ["SUPER_OFFPEAK", "OFFPEAK", "PEAK"]
schedule_op.rates_classes = ["OFFPEAK", "PEAK"]
schedule_flat.rates_classes = ["FLAT"]


def extract_dates(daily):
    return [pd.to_datetime(x[0], "%Y-%m-%d").date() for x in daily.items()]


def category_tally_by_plan(daily=None, plan=None):
    """
    Returns the daily sum of usage for each tou category in a dictionary.
    """
    schedule = rates_schedules[plan]
    return category_tally_by_schedule(daily=daily, schedule=schedule)


def category_tally_by_schedule(daily=None, schedule=None):
    """
    Returns the daily sum of usage for each tou category in a dictionary.
    """
    daily_arrays = {l: np.array([]) for l in schedule.rates_classes}

    for date, consumption_data in daily.items():
        d = pd.to_datetime(date, "%Y-%m-%d").date()

        for category in daily_arrays:
            current_array = daily_arrays[category]
            # hourly data
            if len(consumption_data)==24:
                daily_arrays[category] = np.append(current_array, sum([consumption_data[hour] for hour in schedule(d)[category]]))
            else:
                daily_arrays[category] = np.append(current_array, sum([sum(consumption_data[hour*4:hour*4+4]) for hour in schedule(d)[category]]))
    return daily_arrays


def tou_stacked_plot(daily=None, plan=None):
    """
    Generates a stacked bar plot that shows the decomposed energy usage of each day.
    """
    dates = extract_dates(daily)

    daily_arrays = category_tally_by_plan(daily=daily, plan=plan)

    # plot the daily summary with stacked bars
    plt.figure()

    previous = np.zeros(len(dates))
    for index, category in enumerate(daily_arrays):
        # print(index, category, daily_arrays[category])
        plt.bar(dates, daily_arrays[category], label=category, color=f"C{index}", bottom=previous)
        previous += daily_arrays[category]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.ylabel("Consumption (kWh)")
    plt.grid(linestyle="--", axis="y")
    plt.title("Daily Consumption")
    plt.legend()
    plt.tight_layout()
    plt.show()


def aggregated_hourly_plot(daily=None):
    """
    Generates 24 hour sum of energy usage for each day.
    """
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
    """
    Generate plots for hourly energy usage for each day (one day each row).
    """
    if len(daily.index) >= 50:
        return
    dates = extract_dates(daily)
    fig, axs = plt.subplots(len(daily.index), 1, sharex=True)

    i = 0
    # series can use iteritems method
    for date, consumption_data in daily.items():
        # print(date, consumption_data)
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
    plt.show()


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
    plt.show()


def load_df(filename):
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
    return df


@click.command()
@click.option("-f", "--filename", required=True, help="The full path of the 60-minute exported electricity usage file.")
@click.option("-z", "--zone", default="coastal", show_default=True, help="The climate zone of the house. Should be one of coastal, inland, mountain, desert.")
@click.option("--billing_cycles", default=None, type=int, help="The number of billing cycles. If not provided, will be estimated.")
@click.option("--pcia_year", default=2021, type=int, show_default=True, help="The vantage point of PCIA fee. (indicated on the bill)")
def plot_sdge_hourly(filename, billing_cycles, zone, pcia_year):
    df = load_df(filename)
    # group the hourly data by dates, get a pandas series, which is 1-dimensional with label
    # daily_summary = df.groupby('Date')['Consumption'].sum()

    daily = df.groupby("Date")["Consumption"].apply(tuple)

    # plot the daily summary bar plot without stacking
    # plt.bar(days,daily_summary.values)

    # tou_stacked_plot(daily=daily, plan="TOU-DR1")

    # plot day by day
    # daily_hourly_2d_plot(daily=daily)
    # daily_hourly_3d_plot(daily=daily)

    # plot hourly data summed across days
    # aggregated_hourly_plot(daily=daily)
    if billing_cycles is None:
        billing_cycles = int(round(len(daily) / 30.0))

    plans_and_charges = dict()
    c = SDGECaltulator(daily, zone=zone, pcia_rate=rates["PCIA"][pcia_year], billing_cycles=billing_cycles)
    for plan in ["TOU-DR1", "CCA-TOU-DR1", "EV-TOU-5", "CCA-EV-TOU-5", "EV-TOU-2", "CCA-EV-TOU-2", "TOU-DR2", "CCA-TOU-DR2", "DR", "CCA-DR"]:
        estimated_charge = c.calculate(plan=plan)
        plans_and_charges[plan] = estimated_charge

    for item in sorted(plans_and_charges.items(), key=lambda x: x[1]):
        print(f"{item[0]:<15} ${item[1]:.4f} ${item[1]/c.total_usage:.4f}/kWh")


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")

    # print(get_baseline(zone="coastal", season="summer", service_type="electric", multiplier=1.3, billing_days=29))
    # filename = f"{home_dir}/Downloads/sdge_data/Electric_60_Minute_10-20-2023_11-17-2023_20231130.csv"
    # filename = f"{home_dir}/Downloads/sdge_data/Electric_60_Minute_9-21-2023_10-19-2023_20231130.csv"
    # filename = f"{home_dir}/Downloads/sdge_data/Electric_60_Minute_8-31-2023_9-20-2023_20231130.csv"
    # filename = f"{home_dir}/Downloads/sdge_data/Electric_60_Minute_8-31-2023_11-17-2023_20231130.csv"

    plot_sdge_hourly()
