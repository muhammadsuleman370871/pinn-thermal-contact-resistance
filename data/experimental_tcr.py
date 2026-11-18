"""
Published experimental thermal contact resistance / conductance data
for validation of the PINN framework.

Sources:
1. Neutrium (compilation of published data):
   https://neutrium.net/heat-transfer/thermal-contact-resistance/
2. Fieberg & Kneer (2008), Int J Heat Mass Transfer, 51, 1017-1023
3. Dou et al. (2016), Int J Heat Mass Transfer, 94, 156-163
4. Cooper, Mikic & Yovanovich (1969), Int J Heat Mass Transfer, 12, 279-300
5. Fletcher (1988), J Heat Transfer, 110, 1059-1070
"""

import numpy as np

# ============================================================
# Experimental TCC/TCR data for Aluminum-Steel contacts
# ============================================================
# TCC = thermal contact conductance [W/m²·K]
# TCR = 1/TCC [m²·K/W]

EXPERIMENTAL_DATA = {
    "Al-Steel_ground_smooth": {
        "description": "Aluminum / Steel Ct-30, ground surfaces, Ra = 1.4-2.0 µm, T = 20°C",
        "source": "Neutrium compilation (originally from Soviet-era handbooks)",
        "contact_pressure_atm": [10, 15, 25, 35],
        "TCC_W_m2K": [50000, 52000, 55000, 59000],
        "surface_roughness_um": 1.7,  # average of 1.4-2.0
        "temperature_C": 20,
    },
    "Al-Steel_milled_rough": {
        "description": "Aluminum / Steel Ct-30, milled surfaces, Ra = 4.5-7.2 µm, T = 20°C",
        "source": "Neutrium compilation",
        "contact_pressure_atm": [10, 20, 30],
        "TCC_W_m2K": [4800, 6500, 8300],
        "surface_roughness_um": 5.85,  # average of 4.5-7.2
        "temperature_C": 20,
    },
    "Al-SS_smooth": {
        "description": "Aluminum / Stainless Steel, smooth surfaces, Ra = 1.0-2.0 µm, T = 20°C",
        "source": "Neutrium compilation",
        "contact_pressure_atm": [10, 20],
        "TCC_W_m2K": [16400, 20800],
        "surface_roughness_um": 1.5,
        "temperature_C": 20,
    },
    "Al-SS_rough": {
        "description": "Aluminum / Stainless Steel, rough surfaces, Ra = 20-30 µm, T = 20°C",
        "source": "Neutrium compilation",
        "contact_pressure_atm": [10, 20],
        "TCC_W_m2K": [2900, 3600],
        "surface_roughness_um": 25,
        "temperature_C": 20,
    },
    "Al-Al_ground": {
        "description": "Aluminum / Aluminum, ground surfaces, Ra = 2.54 µm, T = 150 K",
        "source": "Neutrium compilation",
        "contact_pressure_atm": [1.2, 1.8, 2.5],
        "TCC_W_m2K": [11400, 11400, 11400],
        "surface_roughness_um": 2.54,
        "temperature_C": -123,  # 150 K
    },
    "SS416-SS416_ground": {
        "description": "SS 416 / SS 416, ground surfaces, Ra = 2.54 µm, T = 90-200 K",
        "source": "Neutrium compilation",
        "contact_pressure_atm": [0.3, 1.0, 2.5],
        "TCC_W_m2K": [3800, 3800, 3800],
        "surface_roughness_um": 2.54,
        "temperature_C": 145,  # average of 90-200 K range
    },
}


def get_tcr_range_al_steel():
    """
    Return the range of experimentally measured TCR for Al-Steel contacts.

    Returns dict with:
        tcr_min, tcr_max: range in m²·K/W
        tcc_min, tcc_max: range in W/m²·K
        typical_tcr: representative value for moderate contact pressure
    """
    all_tcc = []
    for key, data in EXPERIMENTAL_DATA.items():
        if "Al" in key and ("Steel" in key or "SS" in key):
            all_tcc.extend(data["TCC_W_m2K"])

    tcc_min = min(all_tcc)
    tcc_max = max(all_tcc)
    tcr_min = 1.0 / tcc_max
    tcr_max = 1.0 / tcc_min

    # Typical value: Al-SS smooth at moderate pressure
    typical_tcc = 16400  # W/m²·K at 10 atm, smooth surfaces
    typical_tcr = 1.0 / typical_tcc

    return {
        "tcr_min": tcr_min,
        "tcr_max": tcr_max,
        "tcc_min": tcc_min,
        "tcc_max": tcc_max,
        "typical_tcr": typical_tcr,
        "typical_tcc": typical_tcc,
        "n_datapoints": len(all_tcc),
    }


def validate_against_experimental(Rc_estimated, verbose=True):
    """
    Check whether a PINN-estimated Rc falls within the experimentally
    observed range for Al-Steel contacts.

    Args:
        Rc_estimated: estimated TCR in m²·K/W
        verbose: print comparison

    Returns:
        dict with validation results
    """
    exp = get_tcr_range_al_steel()

    in_range = exp["tcr_min"] <= Rc_estimated <= exp["tcr_max"]

    # Find closest experimental condition
    closest_key = None
    closest_diff = float('inf')
    for key, data in EXPERIMENTAL_DATA.items():
        if "Al" in key and ("Steel" in key or "SS" in key):
            for tcc in data["TCC_W_m2K"]:
                tcr = 1.0 / tcc
                diff = abs(tcr - Rc_estimated)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_key = key

    result = {
        "Rc_estimated": Rc_estimated,
        "in_experimental_range": in_range,
        "experimental_tcr_min": exp["tcr_min"],
        "experimental_tcr_max": exp["tcr_max"],
        "closest_condition": closest_key,
        "closest_description": EXPERIMENTAL_DATA[closest_key]["description"],
    }

    if verbose:
        print("=" * 60)
        print("  Validation Against Experimental Data")
        print("=" * 60)
        print(f"  Estimated Rc:     {Rc_estimated:.4e} m²·K/W")
        print(f"  Experimental range: {exp['tcr_min']:.4e} - {exp['tcr_max']:.4e} m²·K/W")
        print(f"  In range:         {'YES' if in_range else 'NO'}")
        print(f"  Closest match:    {closest_key}")
        print(f"    Description:    {EXPERIMENTAL_DATA[closest_key]['description']}")
        print(f"    Source:         {EXPERIMENTAL_DATA[closest_key]['source']}")

    return result


def print_experimental_summary():
    """Print a formatted summary of all experimental data."""
    print("=" * 80)
    print("  Published Experimental Thermal Contact Conductance Data")
    print("=" * 80)

    for key, data in EXPERIMENTAL_DATA.items():
        print(f"\n  {key}:")
        print(f"    {data['description']}")
        print(f"    Source: {data['source']}")
        print(f"    Roughness: {data['surface_roughness_um']} µm")
        print(f"    {'P (atm)':>10s} {'TCC (W/m²·K)':>15s} {'TCR (m²·K/W)':>15s}")
        for p, tcc in zip(data['contact_pressure_atm'], data['TCC_W_m2K']):
            tcr = 1.0 / tcc
            print(f"    {p:10.1f} {tcc:15.0f} {tcr:15.4e}")


if __name__ == "__main__":
    print_experimental_summary()
    print()

    # Our simulation uses Rc = 1e-4 m²·K/W
    validate_against_experimental(1.0e-4)
