#!/usr/bin/env python3

import numpy as np
import pandas as pd
import datetime
import yaml
import traceback
import os
from functools import cache
from collections import namedtuple
import click
import tempfile
from pypdf import PdfReader, PdfWriter
import pathlib

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
# for 3d bar plot
from mpl_toolkits.mplot3d.axes3d import Axes3D

# for num2date
import matplotlib.dates as mpl_dates

# for FuncFormatter
import matplotlib.ticker as ticker

# for holiday exclusion
from pandas.tseries.holiday import USFederalHolidayCalendar

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

def convert_12h_to_24h(time_str):
    dt = datetime.datetime.strptime(time_str, "%I:%M %p")
    # extract the hour
    time_24h_str = dt.strftime("%H")
    return int(time_24h_str)


def validate_dates(days):
    """
    To validate that the data is within one continuous year.
    """
    # days is sorted from low to high
    if days[0].date.year == days[-1].date.year:
        # all data from the same year
        pass
    if days[-1].date.year - days[0].date.year > 1:
        # this contains data from more than one year
        raise ValueError("Cannot use data from more than one year")
    if days[-1].date.year - days[0].date.year == 1:
        # span year n and year n+1
        if days[-1].date.month > days[0].date.month:
            # this contains data from more than one year
            # for example 2023-09 is more than 1 year from any day in 2022-08
            raise ValueError("Cannot use data from more than one year")
        elif days[-1].date.month == days[0].date.month:
            # starting from (y,m,d), you can get to (y+1,m,d-1) as the last day when d!=1
            if days[-1].date.day >= days[0].date.day:
                raise ValueError("Cannot use data from more than one year")


SDGEDay = namedtuple("SDGEDate", ["date", "season", "daytype"])

pwd = os.path.dirname(os.path.realpath(__file__))


class SDGECaltulator:
    def __init__(self, daily_24h, rates, zone="coastal", service_type="electric", pcia_year="2021", solar="NA"):
        self.daily_24h = daily_24h
        self.days = [SDGEDay(date, get_season(date), get_holiday_status(date)) for date in extract_dates(self.daily_24h)]
        self.zone = zone
        self.rates = rates
        self.pcia_rate = self.rates["PCIA"][int(pcia_year)]
        self.service_type = service_type
        self.total_usage = sum([sum([x[1] for x in usage]) for date, usage in self.daily_24h.items()])
        self.solar = solar
        #assert self.days[0].date.year == self.days[-1].date.year, "all data must be from the same year"
        validate_dates(self.days)
        self.print_info()

    def print_info(self):
        print(f"starting:{self.days[0].date} ending:{self.days[-1].date}")
        print(f"{len(self.days)} days, {len([x for x in self.days if x.season=='summer'])} summer days, {len([x for x in self.days if x.season=='winter'])} winter days")
        if self.solar != "NA":
            print(f"solar setup: {self.solar}")
        summer_usage = sum([sum([x[1] for x in usage]) for date, usage in self.daily_24h.items() if get_season(date)=='summer'])
        winter_usage = sum([sum([x[1] for x in usage]) for date, usage in self.daily_24h.items() if get_season(date)=='winter'])
        print(f"total_usage:{self.total_usage:.4f} kWh (summer: {summer_usage:.4f} kWh winter: {winter_usage:.4f} kWh)")

    def generate_plots(self):
        # plot hourly data summed across days
        aggregated_hourly_net_usage_plot(daily=self.daily_24h)
        daily_net_usage_plot(daily=self.daily_24h)

    @cache
    def tally(self, schedule=None):
        daily_arrays = tou_period_tally_by_schedule(daily=self.daily_24h, schedule=schedule)
        rates_classes = schedule_to_rate_classes_mapping[schedule]

        season_days_counter = {"summer": 0, "winter": 0}
        holiday_days_counter = {"workday":0, "holiday": 0}
        # tally the summer usage and winter usage
        season_class_tally = {"summer": {x: 0.0 for x in rates_classes}, "winter": {x: 0.0 for x in rates_classes}}
        holiday_class_tally = {"workday": {x: 0.0 for x in rates_classes}, "holiday": {x: 0.0 for x in rates_classes}}
        for k, day in enumerate(self.days):
            season_days_counter[day.season] += 1
            holiday_days_counter[day.daytype] += 1
            for rate_class in rates_classes:
                season_class_tally[day.season][rate_class] += daily_arrays[rate_class][k]
                holiday_class_tally[day.daytype][rate_class] += daily_arrays[rate_class][k]
        return rates_classes, season_days_counter, season_class_tally, holiday_days_counter, holiday_class_tally

    def calculate(self, plan=None):
        # usage tally
        rates = self.rates
        rates_classes, season_days_counter, season_class_tally, holiday_days_counter, holiday_class_tally = self.tally(schedule=rates_schedules[plan])

        total_fee = 0.0
        results = dict()

        for season in ["winter", "summer"]:
            season_total_usage = sum(season_class_tally[season].values())
            usage_by_class = season_class_tally[season]
            rates_by_class = rates[plan][season]
            cost_by_class = [usage_by_class[rates_class] * rates_by_class[rates_class] for rates_class in usage_by_class]
            results.setdefault("season_class_cost", {})[season] = cost_by_class

            total_fee += sum(cost_by_class)

            allowance_deduction = get_allowance_deduction(
                zone=self.zone,
                season=season,
                service_type=self.service_type,
                billing_days=season_days_counter[season],
                total_usage=season_total_usage,
                credit_per_kwh=rates[plan][season]["credit"],
            )
            # remove the deduction
            total_fee -= allowance_deduction
            results.setdefault("season_allowance_credit", {})[season] = allowance_deduction
        # apply the recurring service fee
        # SDGE apply month service fee based on days (based on my own plan switching experience)
        service_fee = 0.0
        if "monthly_service_fee" in rates[plan]:
            service_fee = rates[plan]["monthly_service_fee"]/30.0 * len(self.days)
        if "daily_service_fee" in rates[plan]:
            service_fee = rates[plan]["daily_service_fee"] * len(self.days)
        total_fee += service_fee

        # apply the PCIA rates for CCA
        pcia_fee = 0.0
        if "CCA" in plan:
            pcia_fee = self.total_usage * self.pcia_rate
            total_fee += pcia_fee
        results["total_fee"] = total_fee
        results["service_fee"] = service_fee
        results["pcia_fee"] = pcia_fee
        results["season_class_usage"] = season_class_tally
        results["holiday_class_usage"] = holiday_class_tally
        return results


def calculate_misc_fees(total_usage=0.0, pcia_rate=0.01687):
    misc_fee = 0.0

    return misc_fee


@cache
def get_allowance_deduction(zone="coastal", season=None, service_type="electric", billing_days=30, total_usage=0.0, credit_per_kwh=0.11724):
    # calculate 130% allowance deduction
    baseline130 = get_baseline(zone=zone, season=season, service_type=service_type, multiplier=1.3, billing_days=billing_days)
    # for non-solar users, and solar users with net consumption (more consumption than generation)
    if total_usage > 0:
        deducted_usage = min(total_usage, baseline130)
    # for solar users with net generation (more generation than consumption), the credit would be negative
    else:
        deducted_usage = max(total_usage, -baseline130)
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

def get_holiday_status(date):
    weekday = date.weekday()
    # mark US holidays
    holidays = holidays_of_year(date.year)
    if weekday == 5 or weekday == 6 or date in holidays:
        return "holiday"
    return "workday"

schedule_to_rate_classes_mapping = {
    "sop": ["super_offpeak", "offpeak", "peak"],
    "op":  ["offpeak", "peak"],
    "flat": ["flat"]
}

# https://www.sdge.com/regulatory-filing/16026/residential-time-use-periods
def daily_schedule(date, schedule):
    if schedule == "sop":
        if get_holiday_status(date) == "holiday":
            return {"super_offpeak": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, "offpeak": {14, 15, 21, 22, 23}, "peak": {16, 17, 18, 19, 20}}
        return {"super_offpeak": {0, 1, 2, 3, 4, 5, 10, 11, 12, 13}, "offpeak": {6, 7, 8, 9, 14, 15, 21, 22, 23}, "peak": {16, 17, 18, 19, 20}}
    if schedule == "op":
        return {"offpeak": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23}, "peak": {16, 17, 18, 19, 20}}
    if schedule == "flat": 
        return {"flat": {i for i in range(24)}}

@cache
def holidays_of_year(year):
    cal = USFederalHolidayCalendar()
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year + 1, 1, 1)
    holidays = cal.holidays(start=start, end=end).to_pydatetime()
    return holidays

def tou_period_tally_by_plan(daily=None, plan=None):
    """
    Returns the daily sum of usage for each tou_period in a dictionary.
    """
    schedule = rates_schedules[plan]
    return tou_period_tally_by_schedule(daily=daily, schedule=schedule)

def tou_period_tally_by_schedule(daily=None, schedule=None):
    """
    Returns the daily sum of usage for each tou_period in a dictionary.
    """
    daily_arrays = {l: np.array([]) for l in schedule_to_rate_classes_mapping[schedule]}

    for date, consumption_data in daily.items():
        d = pd.to_datetime(date, "%Y-%m-%d").date()

        for tou_period in daily_arrays:
            current_array = daily_arrays[tou_period]
            # remove assumption about number of data items
            daily_arrays[tou_period] = np.append(
                current_array, sum([consumption_data[i][1] for i in range(len(consumption_data)) if consumption_data[i][0] in daily_schedule(d, schedule)[tou_period]])
            )
    return daily_arrays

def load_df(filename):
    # read the csv and skip the first rows
    df = pd.read_csv(
        filename,
        skiprows=13,
        index_col=False,
        usecols=["Date", "Start Time", "Duration", "Consumption", "Net"],
        skipinitialspace=True,
        dtype={"Consumption": np.float32},
        parse_dates=["Date"],
    )
    return df

def extract_dates(daily):
    return [pd.to_datetime(x[0], "%Y-%m-%d").date() for x in daily.items()]

def build_rates_schedules(rates):
    """
    Extract plan to schedule mapping
    """
    global rates_schedules
    rates_schedules = dict()
    for key in rates:
        if key != "PCIA":
            if "super_offpeak" in rates[key]["summer"]:
                rates_schedules[key] = "sop"
            elif "offpeak" in rates[key]["summer"]:
                rates_schedules[key] = "op"
            else:
                rates_schedules[key] = "flat"

def tou_stacked_plot(daily=None, plan=None, plan_rates=None, output_file=None, show=False):
    dates = extract_dates(daily)
    daily_arrays = tou_period_tally_by_plan(daily=daily, plan=plan)

    if plan_rates is None:
        raise ValueError("plan_rates is required")

    daily_cost_arrays = {}
    for category, usage_array in daily_arrays.items():
        daily_cost_arrays[category] = np.array(
            [
                usage * plan_rates[get_season(date)][category]
                for date, usage in zip(dates, usage_array)
            ]
        )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8.5))

    previous_usage = np.zeros(len(dates))
    previous_cost = np.zeros(len(dates))

    for index, category in enumerate(daily_arrays):
        color = f"C{index}"
        label = category.replace("_", " ").title()

        axes[0].bar(dates, daily_arrays[category], label=label, color=color, bottom=previous_usage)
        previous_usage += daily_arrays[category]

        axes[1].bar(dates, daily_cost_arrays[category], label=label, color=color, bottom=previous_cost)
        previous_cost += daily_cost_arrays[category]

    axes[0].set_ylabel("Consumption (kWh)")
    axes[0].set_title(f"{plan} Daily TOU Usage")
    axes[0].grid(linestyle="--", axis="y")
    axes[0].legend()

    axes[1].set_ylabel("Energy Cost ($)")
    axes[1].set_title("Daily Energy Cost Before Credits and Fees")
    axes[1].grid(linestyle="--", axis="y")

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def plot_all_plans_to_pdf(daily=None, rates=None, output_pdf="daily_tou_usage_cost.pdf"):
    plans = [plan for plan in rates if plan != "PCIA"]

    writer = PdfWriter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        for index, plan in enumerate(plans):
            page_pdf = tmp_dir / f"{index:03d}_{plan}.pdf"

            tou_stacked_plot(daily=daily, plan=plan, plan_rates=rates[plan], output_file=page_pdf, show=False)

            reader = PdfReader(str(page_pdf))
            for page in reader.pages:
                writer.add_page(page)

        with open(output_pdf, "wb") as f:
            writer.write(f)


def daily_net_usage_plot(daily=None):
    """
    Generates sum of energy usage for each day.
    """
    dates = extract_dates(daily)
    plt.figure()
    plt.title(f'Daily Net Usage: {dates[0].strftime("%Y/%m/%d")} to {dates[-1].strftime("%Y/%m/%d")}')
    daily_net_usage = [sum(consumption_data)[1] for date, consumption_data in daily.items()]

    plt.bar(dates, daily_net_usage)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.ylabel("Net Usage (kWh)")
    plt.savefig(f"plot_daily_net_usage_{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}.png", dpi=300)


def aggregated_hourly_net_usage_plot(daily=None):
    """
    Generates aggregated usage by hour across all dates.
    """
    dates = extract_dates(daily)
    # plot the hourly summary
    plt.figure()
    plt.title(f'Aggregated Hourly Consumption: {dates[0].strftime("%Y/%m/%d")} to {dates[-1].strftime("%Y/%m/%d")}')
    # handles cases where readings from some hour may be missing
    aggregated_hourly = [sum(chain.from_iterable([[daily[x][k][1] for k in range(len(daily[x])) if daily[x][k][0] == i] for x in daily.index])) for i in range(24)]
    plt.bar(list(range(24)), aggregated_hourly)
    plt.ylabel("Net Usage (kWh)")
    plt.xlabel("Hour")
    plt.xlim([-0.5, 23.5])
    plt.savefig(f"plot_aggregated_hourly_net_usage_{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}.png", dpi=300)


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
    for i, (date, consumption_data) in enumerate(daily.items()):
        pairs = np.asarray(list(consumption_data), dtype=float)

        hours = pairs[:, 0]
        usage = pairs[:, 1]

        axs[i].bar(hours, usage)
        axs[i].set_yticks([])

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

    usage_by_day = []

    for date, consumption_data in daily.items():
        pairs = np.asarray(list(consumption_data), dtype=float)

        # pairs[:, 0] is hour
        # pairs[:, 1] is consumption
        usage = pairs[:, 1]

        usage_by_day.append(usage)

    usage_by_day = np.asarray(usage_by_day, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xvalues = np.arange(24)
    yvalues = np.array([mpl_dates.date2num(d) for d in daily.index])

    xx, yy = np.meshgrid(xvalues, yvalues)

    xx = xx.flatten()
    yy = yy.flatten()

    dz = usage_by_day.flatten()
    zz = np.zeros_like(dz)

    dx = np.ones_like(dz)
    dy = np.ones_like(dz)

    if dz.max() > 0:
        colors = plt.cm.jet(dz / dz.max())
    else:
        colors = plt.cm.jet(np.zeros_like(dz))

    ax.set_xlim([-0.5, 23.5])
    ax.set_ylim([min(yvalues), max(yvalues)])

    ax.set_xlabel("Hour")
    ax.set_zlabel("Consumption (kWh)")

    ax.bar3d(xx, yy, zz, dx, dy, dz, color=colors)

    num2formatted = lambda x, _: mpl_dates.num2date(x).strftime("%Y-%m-%d")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(num2formatted))

    ax.tick_params(axis="y", labelrotation=90)

    plt.title(
        f'Daily Details 3D: {dates[0].strftime("%Y/%m/%d")} '
        f'to {dates[-1].strftime("%Y/%m/%d")}'
    )

    plt.show()

@click.command()
@click.option("-f", "--filename", required=True, help="The full path of the 60-minute exported electricity usage file.")
@click.option("-z", "--zone", default="coastal", type=click.Choice(["coastal", "inland", "mountain", "desert"]), show_default=True, help="The climate zone of the house.")
@click.option("-s", "--solar", default="NA", type=click.Choice(["NA", "NEM1.0"]), show_default=True, help="The solar setup.")
@click.option(
    "--pcia_year", default="2021", type=click.Choice([str(x) for x in range(2009, 2026)]), show_default=True, help="The vintage of the PCIA fee. (indicated on the bill)"
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Show detailed per-plan breakdown.")
def plot_sdge_hourly(filename, zone, pcia_year, solar, verbose):
    df = load_df(filename)

    interval = df.iloc[0]["Duration"]
    # convert the 12h-format start time to 24h-format
    df["Start Time"] = pd.to_datetime(df["Start Time"], format="%I:%M %p").dt.strftime("%H")
    # convert hour to int index
    df["Start Time"] = df["Start Time"].astype(int)

    if solar == "NA":
        consumption_column_label = "Consumption"
    elif solar == "NEM1.0":
        consumption_column_label = "Net"

    # occasionally there are two readings for the same time slot, for now, we sum up the duplicates #TODO: ask SDGE what's happening!
    # df = df.drop_duplicates(subset=["Date","Start Time"], keep="last")
    # this step sums duplicates for 60-min interval data; aggregates the 15-min interval data into hourly data
    df = df.astype("object").groupby(["Date", "Start Time"], as_index=False, sort=False).agg("sum")  # use astype to prevent pd from converting int to float
    daily = df.groupby("Date")[["Start Time", consumption_column_label]].apply(lambda x: tuple(x.values)) # sorted by date by default

    plans_and_charges = dict()
    applied_rates = "sdge_rates_20260601.yaml"
    print(f"The applied rates: {applied_rates}")
    rates_path = os.path.join(pwd, "rates", applied_rates)

    rates = load_yaml(rates_path)
    build_rates_schedules(rates)
    c = SDGECaltulator(daily, rates, zone=zone, pcia_year=pcia_year, solar=solar)

    if solar == "NA":
        plans = [plan for plan in rates if plan not in ["PCIA", "DR-SES", "CCA-DR-SES"]] 
    else:
        plans = [plan for plan in rates if plan not in ["PCIA", "DR", "CCA-DR"]] 

    for plan in plans:
        plans_and_charges[plan] = c.calculate(plan=plan)


    sorted_plans_and_charges = sorted(plans_and_charges.items(), key=lambda x: x[1]["total_fee"])

    for item in sorted_plans_and_charges:
        print(f"{item[0]:<15} ${item[1]['total_fee']:.4f} ${item[1]['total_fee']/c.total_usage:.4f}/kWh")

    if verbose:
        for plan, charges in sorted_plans_and_charges:
            schedule = rates_schedules[plan]
            rate_classes = schedule_to_rate_classes_mapping[schedule]
            print("")
            print(f"{plan}")
            print("-" * len(plan))
            for season in ["winter", "summer"]:
                usage_by_class = charges["season_class_usage"][season]
                cost_by_class = charges["season_class_cost"][season]
                season_usage = sum(usage_by_class.values())
                season_cost = sum(cost_by_class)
                print(f"{season}: {season_usage:.4f} kWh, ${season_cost:.4f}")
                for rate_class, cost in zip(rate_classes, cost_by_class):
                    usage = usage_by_class[rate_class]
                    rate = rates[plan][season][rate_class]
                    print(f"  {rate_class:<15} {usage:>10.4f} kWh x ${rate:.5f}/kWh = ${cost:.4f}")
                print(f"  allowance credit: -${charges['season_allowance_credit'][season]:.4f}")
            print(f"service fee: ${charges['service_fee']:.4f}")
            if charges["pcia_fee"]:
                print(f"PCIA: ${charges['pcia_fee']:.4f}")
            print(f"total: ${charges['total_fee']:.4f}")

    #tou_stacked_plot(daily=daily, plan="TOU-DR1", plan_rates=rates[plan])

    # plot day by day
    #daily_hourly_2d_plot(daily=daily)
    #daily_hourly_3d_plot(daily=daily)

    #c.generate_plots()
    #plot_all_plans_to_pdf(daily=daily, rates=rates, output_pdf="daily_tou_usage_cost_all_plans.pdf")

if __name__ == "__main__":
    # print(get_baseline(zone="coastal", season="summer", service_type="electric", multiplier=1.3, billing_days=29))

    plot_sdge_hourly()
