# my_sdge
Compare different popular SDGE residential rates plans, using the Green Button Download exported usage file available from the SDGE portal.

- electricity only (I only have experience with electricity only place)
- no solar (I am a tenant at a place without solar)
- compares: `TOU-DR1`, `TOU-DR2`, `EV-TOU-5`, `EV-TOU-2`, `DR`
- supports SDGE generation + SDGE delivery (represented with plan name) and CCA generation + SDGE delivery (represented with prefix CCA + plan name)
- calculates baseline allowance credit when applicable.
- TAXES & FEES are not included. PCIA is included for CCA option calculation.

## Install
```bash
git clone https://github.com/corruptbear/my_sdge
cd my_sdge
python3.9 -m pip install -r requirements.txt
```

Should work with both python3.9 and python3.10. Other versions are not tested.

## Usage
see the help:
```bash
python3.9 sdge_hourly.py --help
```
```
Usage: sdge_hourly.py [OPTIONS]

Options:
  -f, --filename TEXT       The full path of the 60-minute exported
                            electricity usage file.  [required]
  -z, --zone TEXT           The climate zone of the house. Should be one of
                            coastal, inland, mountain, desert.  [default:
                            coastal]
  --billing_cycles INTEGER  The number of billing cycles. If not provided,
                            will be estimated.
  --pcia_year INTEGER       The vantage point of PCIA fee. (indicated on the
                            bill)  [default: 2021]
  --help                    Show this message and exit.
```

## Example
An example 60-min resolution usage file (historical data) is provided.

To use the historical data to compare different plans using the current rates:

```bash
# ensure that you are currently in the downloaded repo folder
python3.9 sdge_hourly.py -f Electric_60_Minute_11-1-2022_11-30-2022_20230819.csv -z coastal --billing_cycles 1 --pcia_year 2021
```
Outputs (the plans are ranked from lowest cost to highest cost):
```
starting:2022-11-01 ending:2022-11-30
30 days, 0 summer days, 30 winter days
total_usage:817.4150 kWh
number of billing cycles:1
CCA-EV-TOU-5    $308.8959 $0.3779/kWh
EV-TOU-5        $313.9536 $0.3841/kWh
CCA-EV-TOU-2    $336.4075 $0.4116/kWh
EV-TOU-2        $341.4653 $0.4177/kWh
CCA-TOU-DR2     $415.5525 $0.5084/kWh
CCA-TOU-DR1     $416.1811 $0.5091/kWh
TOU-DR2         $419.5327 $0.5132/kWh
TOU-DR1         $420.1850 $0.5140/kWh
CCA-DR          $421.2420 $0.5153/kWh
DR              $425.4598 $0.5205/kWh
```

## FAQ

### Where to get the 60-minute usage csv file?
- Using a desktop computer, sign into your SDGE account.
- Go to https://myaccount.sdge.com/portal/Usage/Index
- Click the `Green Button Download` icon.
- Select the starting date and the ending date (better aligned with real billing cycles).

### Can I use a csv file with continuous data from more than one billing cycles?
Yes. 

Some plans are advantageous in the summer, while others are advantageous in the winter, with a usage file covering more months you can then compare the overall costs of different plans over longer period. 

Just make sure that the starting date and the ending date are in the same year. 

And don't forget to indicate the number of billing cycles using the `--billing_cycles` command line option.

### How can I find the PCIA vantage point?
On your PDF SDGE bill, the vantage point year for your PCIA fee is indicated.

For example, on my bill, in the section above "Total Electric Charges", "PCIA 2021" is listed, which means the vantage point is 2021.
