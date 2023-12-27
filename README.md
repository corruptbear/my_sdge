# my_sdge
Compare different popular SDGE residential rates plans, using the Green Button Download exported usage file available from the SDGE portal.

- electricity only (I only have experience with electricity only place)
- supports non-solar (default) and NEM1.0
- supports both 15-min resolution data and 60-min resolution data
- compares: `TOU-DR1`, `TOU-DR2`, `EV-TOU-5`, `EV-TOU-2`, `DR`(for non-solar user), `DR-SES`(for NEM1.0 user)
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
  -f, --filename TEXT             The full path of the 60-minute exported
                                  electricity usage file.  [required]
  -z, --zone [coastal|inland|mountain|desert]
                                  The climate zone of the house.  [default:
                                  coastal]
  -s, --solar [NA|NEM1.0]         The solar setup.  [default: NA]
  --pcia_year [2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023]
                                  The vintage of the PCIA fee. (indicated on
                                  the bill)  [default: 2021]
  --help                          Show this message and exit.
```

## Example
An example 60-min resolution usage file (historical data) is provided.

To use the historical data to compare different plans using the current rates:

```bash
# ensure that you are currently in the downloaded repo folder
python3.9 sdge_hourly.py -f Electric_60_Minute_11-1-2022_11-30-2022_20230819.csv -z coastal --pcia_year 2021
```
Outputs (the plans are ranked from lowest cost to highest cost):
```
starting:2022-11-01 ending:2022-11-30
30 days, 0 summer days, 30 winter days
total_usage:817.4150 kWh
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

If you are a NEM1.0 user, add `-s NEM1.0` to the end of the command.

## FAQ

### Where to get the 15-min or 60-minute usage csv file?
- Using a desktop computer, sign into your SDGE account.
- Go to https://myaccount.sdge.com/portal/Usage/Index
- Click the `Green Button Download` icon.
- Select the starting date and the ending date.

### Can I use a csv file with continuous data from more than one billing cycles?
Yes. 

Some plans are advantageous in the summer, while others are advantageous in the winter, with a usage file covering more months you can then compare the overall costs of different plans over longer period. 

Just make sure that the starting date and the ending date are in the same year. 

### Which climate zone should I use?
Please check [SDGE's arcgis map](https://sempra.maps.arcgis.com/apps/Embed/index.html?webmap=9c7f4ff6255946d7a86d6fca6934db40&extent=-118.0874,32.5219,-115.6731,33.5248&home=true&zoom=true&scale=true&search=true&searchextent=true&disable_scroll=false&theme=light) : put your address into the search box, click `Enter`, then see the result.

### How can I find the PCIA year?
On your PDF SDGE bill, the year of your PCIA fee is indicated.

For example, on my bill, in the section above "Total Electric Charges", "PCIA 2021" is listed, which means the year is 2021.

As different years have different PCIA fee, if the fee for a particular year is high, it might make the usually slightly cheaper CCA plans slightly more expensive than non-CCA.

### How are the bills calculated?

non-CCA: SDGE generation fee + SDGE delivery fee - allowance credit

CCA: CCA generation fee + SDGE delivery fee - allowance credit + PCIA

NEM1.0: similar to above; uses net consumption instead of consumption for calculations; applies the allowance credit as described in [page 6](https://www.sdge.com/sites/default/files/elec_elec-scheds_nem.pdf).
(the current calculation is estimate only, it has not taken non-bypassable into account yet, so it's NEM1.0)