#!/usr/bin/env python3
"""
Extract rate schedules from SDGE PDF files and output YAML format.
"""

import pathlib
import re
import sys
from typing import Dict, List, Tuple
import os
from datetime import datetime

import click
import yaml

try:
    from pypdf import PdfReader
except ImportError:
    print("ERROR: pypdf is required. Install with: pip install pypdf")
    sys.exit(1)

def extract_effective_date_key(text: str) -> str:
    """
    Effective February 1, 2025 -> 20250201
    Effective 6/1/2026 -> 20260601
    """
    match = re.search(
        r"\bEffective\s+("
        r"\d{1,2}/\d{1,2}/\d{4}"
        r"|"
        r"[A-Za-z]+\s+\d{1,2},\s+\d{4}"
        r")\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not find effective date")

    return format_effective_date_key(match.group(1))


def format_effective_date_key(date_text: str) -> str:
    for date_format in ("%m/%d/%Y", "%B %d, %Y"):
        try:
            date = datetime.strptime(date_text, date_format)
            return date.strftime("%Y%m%d")
        except ValueError:
            pass

    raise ValueError(f"Could not parse effective date: {date_text}")


def extract_text_with_layout(pdf_path: pathlib.Path) -> str:
    """Extract text from PDF preserving layout."""
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text(extraction_mode="layout")
    return text

def extract_sdge_pcia(text: str) -> dict[str, dict[int, float]]:
    rates = {}

    for line in text.splitlines():
        match = re.search(
            r"^\s*(\d{4})\s+Vintage\s+\$?([0-9.]+)\s*$",
            line,
        )
        if not match:
            continue

        year = int(match.group(1))
        value = float(match.group(2))

        # Optional: skip old legacy vintage rows if you only want 2009+
        if year >= 2009:
            rates[year] = value

    if not rates:
        raise ValueError("Could not find PCIA rates")

    return rates

def extract_base_service_charge(text: str) -> float:
    match = re.search(
        r"^\s*Base Services Charge\s+\(\$/Day\)\s+(.*)$",
        text,
        flags=re.MULTILINE,
    )

    if not match:
        return 0.0

    row = match.group(1)
    numbers = re.findall(r"-?\d+\.\d+", row)

    if not numbers:
        raise ValueError("No numbers found on Base Services Charge row")

    return float(numbers[-1])

def parse_sdge_schedule(text: str) -> Dict:
    results = {}
    sections = extract_sections(text)
    for schedule_name, section_text in sections:
        schedule_type = determine_type(section_text)
        rates = parse_rates(section_text, schedule_type)
        results[schedule_name] = {**rates}
    return results

def parse_cca_schedule(text: str) -> Dict:
    """Parse CCA PowerOn rates from PDF text.

    Returns dict mapping schedule_name to rates data.
    """
    # Mapping from CCA schedule names to SDGE plan names
    cca_to_sdge_mapping = {
        "TOU-DR-1": "TOU-DR1",
        "TOU-DR-2": "TOU-DR2",
    }

    results = {}
    lines = text.splitlines()

    residential_start = 0
    for i, line in enumerate(lines):
        if "Residential Rate Schedule" in line:
            residential_start = i
            break

    schedule_pattern = re.compile(
        r"\b("
        r"DR-LI-MB|DR-SES|EV-TOU-\d+|EV-TOU|TOU-DR-\d+|TOU-DR|TOU-ELEC|DR|LS"
        r")\b\s{2,}[A-Z]"
    )
    schedule_headers = []
    for i, line in enumerate(lines[residential_start:], start=residential_start):
        schedule_match = schedule_pattern.search(line)
        if schedule_match:
            schedule_headers.append((i, schedule_match.group(1)))

    for header_index, (line_index, schedule_name) in enumerate(schedule_headers):
        next_line_index = (
            schedule_headers[header_index + 1][0]
            if header_index + 1 < len(schedule_headers)
            else len(lines)
        )
        schedule_data = {"summer": {}, "winter": {}}

        for data_line in lines[line_index + 1:next_line_index]:
            # Format: "Season  ChargType  TOU_Period  $PowerOn  $PowerBase"
            # Look for lines with Summer/Winter and price values
            if ("Summer" in data_line or "Winter" in data_line) and "Generation" in data_line:
                # Extract season
                season = "summer" if "Summer" in data_line else "winter"

                # Extract prices (PowerOn is first, PowerBase is second)
                prices = re.findall(r'\$(\d+\.\d+)', data_line)

                if len(prices) >= 2:
                    poweron_rate = float(prices[0])

                    # Determine TOU period type
                    if "On-Peak" in data_line and "Super Off-Peak" not in data_line:
                        schedule_data[season]["peak"] = poweron_rate
                    elif "Off-Peak" in data_line and "Super Off-Peak" not in data_line:
                        schedule_data[season]["offpeak"] = poweron_rate
                    elif "Super Off-Peak" in data_line:
                        schedule_data[season]["super_offpeak"] = poweron_rate
                    elif "Total" in data_line:
                        # Flat rate
                        schedule_data[season]["flat"] = poweron_rate

        # Only add if we found actual rate data
        has_data = any(
            len(schedule_data[season]) > 0 for season in ["summer", "winter"]
        )
        if has_data:
            # Apply mapping to match SDGE plan names
            mapped_name = cca_to_sdge_mapping.get(schedule_name, schedule_name)
            results[mapped_name] = schedule_data

    return results


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Extract sections for each schedule in the PDF text.
    Returns list of (schedule_name, section_text)."""
    sections = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for "SCHEDULE " prefix (uppercase)
        if line.startswith("SCHEDULE "):
            # Extract schedule name: "SCHEDULE EV-TOU" or "SCHEDULE EV-TOU-2"
            # The name may have extra spaces, but we take the next token
            parts = line.split()
            if len(parts) >= 2:
                schedule_name = parts[1]
                # Gather section text from this line until next SCHEDULE or end
                section_lines = []
                j = i
                while j < len(lines) and (
                    j == i or not lines[j].strip().startswith("SCHEDULE ")
                ):
                    section_lines.append(lines[j])
                    j += 1
                section_text = "\n".join(section_lines)
                sections.append((schedule_name, section_text))
                i = j - 1  # will be incremented
        i += 1
    return sections


def determine_type(text: str) -> str:
    """Determine schedule type: tier, sop, or op.
    Rules: if contains tier -> tier, elif contains super_offpeak -> sop, else op."""
    if re.search(r"\bTier\s+[12]\b", text, re.IGNORECASE):
        return "flat"
    if re.search(r"Super Off-Peak", text, re.IGNORECASE):
        return "sop"
    # Default to op (offpeak/onpeak only)
    return "op"


def parse_rates(text: str, schedule_type: str) -> Dict:
    """Parse rates from PDF text."""
    # Split into lines
    lines = text.splitlines()
    # Find Summer and Winter sections
    summer_start = None
    winter_start = None
    for i, line in enumerate(lines):
        if "Summer" in line and summer_start is None:
            summer_start = i
        if "Winter" in line and winter_start is None:
            winter_start = i
    # If not found, try case-insensitive
    if summer_start is None:
        for i, line in enumerate(lines):
            if "summer" in line.lower():
                summer_start = i
                break
    if winter_start is None:
        for i, line in enumerate(lines):
            if "winter" in line.lower():
                winter_start = i
                break
    # Extract sections
    summer_text = (
        "\n".join(lines[summer_start:winter_start])
        if summer_start is not None and winter_start is not None
        else ""
    )
    winter_text = "\n".join(lines[winter_start:]) if winter_start is not None else ""

    # Parse each section
    summer_rates = parse_section(summer_text, schedule_type, "summer")
    winter_rates = parse_section(winter_text, schedule_type, "winter")
    rates = {"summer": summer_rates, "winter": winter_rates}

    # Process based on type
    if schedule_type == "flat":
        # Convert tier to flat with baseline_adjustment_credit
        rates = process_tier_to_flat(rates)
    rates["daily_service_fee"] = extract_base_service_charge(winter_text)

    return rates


def parse_numeric_token(token: str) -> float:
    """Convert token to float, extract the adjustment credit from ()."""
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        return float(token[1:-1])
    else:
        return float(token)

def extract_numeric_tokens(line: str) -> List:
    tokens = line.split()
    numeric_tokens = []
    for token in tokens:
        if re.match(
            r"^\(?\d+\.?\d*\)?$", token.replace("(", "").replace(")", "")
        ):
            numeric_tokens.append(token)
    return numeric_tokens

def parse_section(section_text: str, schedule_type: str, season: str) -> Dict:
    """Parse a season section."""
    lines = section_text.splitlines()
    rates = {"credit": 0.0}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for Baseline Adjustment Credit
        if "Baseline Adjustment Credit" in line:
            # Extract numeric tokens similarly
            tokens = line.split()
            numeric_tokens = extract_numeric_tokens(line)
            if len(numeric_tokens) >= 2:
                credit_adjustment_token = numeric_tokens[-1]
                try:
                    credit = parse_numeric_token(credit_adjustment_token)
                    rates["credit"] = round(credit, 5)
                except ValueError:
                    pass
            continue

        # Determine what's the line is about
        subtype = None
        if schedule_type == "flat":
            if "Tier 1" in line:
                subtype = "tier1"
            elif "Tier 2" in line:
                subtype = "tier2"
        elif schedule_type == "sop":
            # Check for TOU subtypes with Super Off-Peak
            line_stripped = line.lstrip()
            if line_stripped.startswith("Super Off-Peak"):
                subtype = "super_offpeak"
            elif line_stripped.startswith("Off-Peak"):
                subtype = "offpeak"
            elif line_stripped.startswith("On-Peak"):
                subtype = "peak"
        else:  # op
            # Only Off-Peak and On-Peak
            line_stripped = line.lstrip()
            if line_stripped.startswith("Off-Peak"):
                subtype = "offpeak"
            elif line_stripped.startswith("On-Peak"):
                subtype = "peak"

        if not subtype:
            continue

        numeric_tokens = extract_numeric_tokens(line)

        # Need at least two numeric tokens (EECC and Total)
        if len(numeric_tokens) < 2:
            continue

        # Last numeric token is Total, second last is EECC
        total_token = numeric_tokens[-1]
        eecc_token = numeric_tokens[-2]

        try:
            total = parse_numeric_token(total_token)
            eecc = parse_numeric_token(eecc_token)
            rates[subtype] = {"total": round(total, 5), "eecc": round(eecc, 5)}
        except ValueError:
            continue

    return rates


def process_tier_to_flat(rates: Dict) -> Dict:
    """Convert tier schedule to flat schedule with baseline_adjustment_credit.

    For tier schedules:
    - Calculate baseline_adjustment_credit = tier1_total - tier2_total for each season
    - Remove tier1
    - Rename tier2 to flat
    """
    result = {}

    for season, season_data in rates.items():
        # Check if we have tier1 and tier2
        if "tier1" in season_data and "tier2" in season_data:
            tier1 = season_data["tier1"]
            tier2 = season_data["tier2"]

            # Create flat rate from tier2
            result[season] = {
                "flat": {"total": tier2["total"], "eecc": tier2["eecc"]},
                "credit": round(tier2["total"]-tier1["total"], 5)
            }
        else:
            # If not a proper tier schedule, keep as-is
            result[season] = season_data

    return result


def merge_rates(sdge_schedule, cca_schedule, digits=5):
    merged = {}
    periods = ["super_offpeak", "offpeak", "peak", "flat"]

    for plan, sdge_plan in sdge_schedule.items():
        if plan == "PCIA":
            merged["PCIA"] = sdge_plan
            continue

        merged[plan] = {}

        for season in ("summer", "winter"):
            merged[plan][season] = {}

            for period in periods:
                if period in sdge_plan[season]:
                    merged[plan][season][period] = round(
                        sdge_plan[season][period]["total"],
                        digits,
                    )

            merged[plan][season]["credit"] = sdge_plan[season].get("credit", 0.0)

        merged[plan]["daily_service_fee"] = sdge_plan.get("daily_service_fee", 0.0)

        if plan not in cca_schedule:
            continue

        cca_name = f"CCA-{plan}"
        merged[cca_name] = {}

        for season in ("summer", "winter"):
            merged[cca_name][season] = {}

            for period in periods:
                if period not in sdge_plan[season]:
                    continue

                sdge_total = sdge_plan[season][period]["total"]
                sdge_eecc = sdge_plan[season][period]["eecc"]
                cca_rate = cca_schedule[plan][season][period]

                merged[cca_name][season][period] = round(
                    sdge_total - sdge_eecc + cca_rate,
                    digits,
                )

            merged[cca_name][season]["credit"] = sdge_plan[season].get("credit", 0.0)

        merged[cca_name]["daily_service_fee"] = sdge_plan.get("daily_service_fee", 0.0)

    return merged

def process_pdf(pdf_path: pathlib.Path) -> Tuple[str, Dict]:
    """Process a single PDF file, possibly containing multiple schedules.
    Returns dict mapping schedule_name to data."""
    text = extract_text_with_layout(pdf_path)
    if not text:
        print(f"WARNING: Could not extract text from {pdf_path}")
        return "", {}
    rate_publishing_date = extract_effective_date_key(text)
    # CCA file
    if "EECC" not in text:
        return "CCA", rate_publishing_date, parse_cca_schedule(text)
    # SDGE file
    if "EECC" in text:
        return "SDGE", rate_publishing_date, parse_sdge_schedule(text)

def extract_rates(pdf_files):
    sdge_rates = {}
    cca_rates = {}
    sdge_rate_publishing_dates = []
    sdge_example = None
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        filetype, rate_publishing_date, result = process_pdf(pdf_file)
        if filetype == "SDGE":
            sdge_rate_publishing_dates.append(rate_publishing_date)
            sdge_rates.update(result)
            sdge_example = pdf_file
        if filetype == "CCA":
            cca_rates = result

    if len(set(sdge_rate_publishing_dates)) == 0:
        print("ERROR: you need SDGE rate pdf files")
        sys.exit(1)

    if len(set(sdge_rate_publishing_dates)) > 1:
        print("ERROR: not all SDGE rate files are from the same effective date")
        sys.exit(1)

    ## add the PCIA info
    sdge_rates["PCIA"] = extract_sdge_pcia(extract_text_with_layout(sdge_example))
    if len(cca_rates) > 0:
        final_rates = merge_rates(sdge_rates, cca_rates)
    else:
        final_rates = sdge_rates

    #print(*sdge_rates, sep="\n")
    #print(*cca_rates, sep="\n")
    #print(*final_rates,  sep="\n")

    return sdge_rate_publishing_dates[0], final_rates

@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path))
def main(directory):
    output_dir = pathlib.Path(__file__).resolve().parent

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        click.echo(f"No PDF files found in {directory}")
        return

    rate_publishing_date, all_rates = extract_rates(pdf_files)

    # Write YAML
    output = output_dir / f"sdge_rates_{rate_publishing_date}.yaml"
    with open(output, "w") as f:
        yaml.dump(all_rates, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Extracted {len(all_rates)} schedules")
    click.echo(f"Saved rate YAML to {output.resolve()}")


if __name__ == "__main__":
    main()
