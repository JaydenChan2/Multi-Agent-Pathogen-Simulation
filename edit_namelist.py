#!/usr/bin/env python3

import sys
import csv
import os

# ============================================================
# INPUT ARGUMENTS
# ============================================================

nml_path     = sys.argv[1]
start_date   = sys.argv[2]
member_name  = sys.argv[3]
beta_scale   = float(sys.argv[4])
ie_scale     = float(sys.argv[5])
ENABLE_AGE_ADJUSTMENT = True

# ============================================================
# ============================================================
# OPTIONAL ARGUMENTS
# ============================================================

override_filename = (
    sys.argv[6]
    if len(sys.argv) >= 7
    else "maps_variant_member_overrides.csv"
)

parameter_mode = (
    sys.argv[7].upper()
    if len(sys.argv) >= 8
    else "OVERRIDE"
)

if parameter_mode == "GLOBAL":
    use_database = False
    use_overrides = False

elif parameter_mode == "DATABASE":
    use_database = True
    use_overrides = False

elif parameter_mode == "OVERRIDE":
    use_database = True
    use_overrides = True

else:
    print(f"ERROR: unknown parameter_mode = {parameter_mode}")
    sys.exit(1)

print()
print("parameter_mode =", parameter_mode)
print("override_filename =", override_filename)
print("use_database =", use_database)
print("use_overrides =", use_overrides)
print()

month_str, day_str, year_str = start_date.split("-")

month = int(month_str)
day   = int(day_str)
year  = int(year_str)

date_tag_us = f"{month:02d}_{day:02d}_{year}"
date_tag_hy = f"{month:02d}-{day:02d}-{year}"

forecast_dir = f"MAPS_FORECASTS/forecasts/forecast_{date_tag_hy}"
member_dir   = f"{forecast_dir}/{member_name}"

outdir  = member_dir + "/"
expname = f"{member_name.split('_')[0]}_{date_tag_hy}"

ic_filename = f"MAPS_FORECASTS/initial_conditions/maps_IC_{date_tag_us}.nc"

variant_file = (
    f"MAPS_FORECASTS/initial_conditions/"
    f"maps_active_variants_{date_tag_hy}.csv"
)

variant_db_file = (
    "MAPS_FORECASTS/variant_database/"
    "maps_variant_parameter_database.csv"
)

MIN_DB_NMATCH = 30

def load_variant_family_tree(filename):
    """
    Load the phylogenetic classification of pathogen variants from a CSV file.

    Each row maps a variant name to its position in the phylogeny hierarchy
    (family > branch > clade). This tree is later used by
    build_cross_immunity_matrix to assign cross-immunity levels based on how
    closely related two variants are.

    Expected CSV columns: variant, family, branch, clade

    Args:
        filename (str): Path to the variant family tree CSV file.

    Returns:
        dict: Mapping of variant name (str) to a dict with keys
              "family", "branch", and "clade" (all str). Returns an empty
              dict if the file does not exist.
    """
    tree = {}

    if not os.path.exists(filename):
        print(f"Family tree file not found: {filename}")
        return tree

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            variant = row["variant"].strip()

            tree[variant] = {
                "family": row["family"].strip(),
                "branch": row["branch"].strip(),
                "clade": row["clade"].strip()
            }

    print(f"Loaded phylogeny tree: {len(tree)} variants")

    return tree

def build_cross_immunity_matrix(active_variants, family_tree):
    """
    Build an NxN cross-immunity matrix for all active variants.

    Cross-immunity represents how much prior infection with variant i protects
    against variant j. Values are assigned by phylogenetic proximity:

        - Same variant (i == j):         0.90  (near-complete self-immunity)
        - Same family, different variant: 0.75  (closely related)
        - Same branch:                    0.55  (moderately related)
        - Same clade:                     0.35  (distantly related)
        - No shared classification:       0.20  (baseline cross-immunity)

    Variants not found in family_tree are treated as unrelated (0.20 off-
    diagonal). Phylogeny lookup uses find_phylogeny, which walks up the
    variant name hierarchy to find the closest registered ancestor.

    Args:
        active_variants (list[str]): Ordered list of variant labels for the
                                     current forecast.
        family_tree (dict): Output of load_variant_family_tree — maps variant
                            names to their phylogenetic classification.

    Returns:
        list[list[float]]: NxN nested list where matrix[i][j] is the
                           cross-immunity factor from variant i to variant j.
    """
    nv = len(active_variants)
    

    matrix = [[0.20 for _ in range(nv)] for _ in range(nv)]

    for i, vi in enumerate(active_variants):

        for j, vj in enumerate(active_variants):

            if i == j:
                matrix[i][j] = 0.90
                continue
            ai = find_phylogeny(vi,family_tree)
            aj = find_phylogeny(vj, family_tree)

            if ai is None or aj is None:
                matrix[i][j] = 0.20
                continue

            if ai["family"] == aj["family"]:
                matrix[i][j] = 0.75

            elif ai["branch"] == aj["branch"]:
                matrix[i][j] = 0.55

            elif ai["clade"] == aj["clade"]:
                matrix[i][j] = 0.35

            else:
                matrix[i][j] = 0.20

    return matrix

def find_phylogeny(variant, family_tree):
    """
    Look up a variant's phylogenetic classification with hierarchical fallback.

    Variant names follow a dot-delimited hierarchy (e.g. "XBB.1.5.1"). If the
    exact name is not in family_tree, this function strips the trailing
    sub-lineage suffix and retries — repeating until a match is found or the
    name can no longer be shortened. This lets newer sub-variants inherit the
    classification of their registered ancestor.

    Args:
        variant (str): Variant label to look up (e.g. "XBB.1.5.1").
        family_tree (dict): Phylogeny dict produced by load_variant_family_tree.

    Returns:
        dict | None: Classification dict with keys "family", "branch", "clade"
                     if any ancestor is found; None if no match exists at any
                     level of the hierarchy.
    """
    test = variant

    while True:

        if test in family_tree:
            return family_tree[test]

        if "." not in test:
            break

        test = ".".join(test.split(".")[:-1])

    return None

family_tree_file = (
    "MAPS_FORECASTS/variant_database/variant_family_tree.csv"
)

family_tree = load_variant_family_tree(family_tree_file)
use_phylogeny = True

# ============================================================
# READ VARIANT PARAMETER DATABASE
# ============================================================

variant_db = {}

if use_database:

    if os.path.exists(variant_db_file):

        with open(variant_db_file, "r", newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                variant = row.get("variant", "").strip()

                if variant == "":
                    continue

                try:
                    nmatch = int(float(row.get("nmatch", "0")))
                except:
                    continue

                if nmatch < MIN_DB_NMATCH:
                    continue

                try:
                    beta_mult = float(row["beta_multiplier"])
                    ie_mult   = float(row["IE_multiplier"])
                except:
                    continue

                variant_db[variant] = {
                    "beta_multiplier": beta_mult,
                    "IE_multiplier": ie_mult,
                    "nmatch": nmatch
                }

        print(
            f"Variant database loaded: "
            f"{len(variant_db)} variants"
        )

    else:

        print(
            f"Variant database not found: "
            f"{variant_db_file}"
        )

else:

    print("Database disabled")

# ============================================================
# READ VARIANT OVERRIDES
# ============================================================

variant_overrides = {}

if use_overrides:

    variant_override_file = (
        "MAPS_FORECASTS/variant_database/"
        + override_filename
    )

    if os.path.exists(variant_override_file):

        with open(variant_override_file, "r", newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                mid = row.get("member_id", "").strip()
                var = row.get("variant", "").strip()

                if mid == "" or var == "":
                    continue

                try:
                    bmult = float(row["beta_multiplier"])
                    imult = float(row["IE_multiplier"])
                except:
                    continue

                variant_overrides[(mid, var)] = {
                    "beta_multiplier": bmult,
                    "IE_multiplier": imult
                }

        print(
            f"Variant override file loaded: "
            f"{variant_override_file}"
        )

        print(
            f"Member-specific overrides: "
            f"{len(variant_overrides)}"
        )

    else:

        print(
            f"No variant override file found: "
            f"{variant_override_file}"
        )

else:

    print("Overrides disabled")
    
# ============================================================
# READ ACTIVE VARIANT FILE
# ============================================================

variant_labels = []
beta0_values = []
ie0_values = []

if not os.path.exists(variant_file):
    print(f"ERROR: active variant file not found: {variant_file}")
    sys.exit(2)

with open(variant_file, "r", newline="") as f:
    reader = csv.reader(f)

    header = next(reader, None)

    for row in reader:
        if len(row) < 4:
            continue

        label = row[0].strip()

        if label == "":
            continue

        beta_guess = float(row[3])
        try:
            mean_age_days = float(row[4])
        except:
            mean_age_days = 999.0

            # ----------------------------------------------------
        # First-guess MAPS formulas
        # ----------------------------------------------------

        beta0_base = 0.23 + 0.20 * beta_guess
        ie0_base   = 1.00 - 0.25 * beta_guess

        # ----------------------------------------------------
        # Choose multipliers
        # ----------------------------------------------------

        override_key = (member_name.split("_")[0], label)

        #
        # Priority:
        # OVERRIDE
        # DATABASE
        # GLOBAL
        #

        if use_overrides and override_key in variant_overrides:

            beta_mult = (
                variant_overrides[override_key]
                ["beta_multiplier"]
            )

            ie_mult = (
                variant_overrides[override_key]
                ["IE_multiplier"]
            )

            source = "OVERRIDE"

        elif use_database and label in variant_db:

            beta_mult = (
                variant_db[label]
                ["beta_multiplier"]
            )

            ie_mult = (
                variant_db[label]
                ["IE_multiplier"]
            )

            source = "DATABASE"

        else:

            beta_mult = beta_scale
            ie_mult   = ie_scale
            source = "GLOBAL"
        #
        # Growth-period beta guardrail
        #

        if parameter_mode in ["DATABASE", "OVERRIDE"]:
            beta_mult = min(beta_mult, 1.5)

        #
        # Age-aware adjustment
        #

        if ENABLE_AGE_ADJUSTMENT and mean_age_days < 30.0:

            age_factor = mean_age_days / 30.0

            #
            # Cap beta inflation
            #

            beta_mult = min(beta_mult, 1.50)

            #
            # Push toward immune escape
            #

            ie_cap = 0.40 + 0.60 * age_factor

            ie_mult = min(ie_mult, ie_cap)

            age_flag = "YOUNG"

        else:

            age_flag = "MATURE"

        print(
            f"{label:20s} "
            f"age={mean_age_days:6.1f} "
            f"{age_flag:6s} "
            f"source={source:8s} "
        )

        #
        # Compute final values
        #

        beta0 = beta0_base * beta_mult
        ie0   = ie0_base   * ie_mult

        variant_labels.append(label)
        beta0_values.append(beta0)
        ie0_values.append(ie0)
        print(
            f"{label:20s} "
            f"source={source:8s} "
            f"beta_guess={beta_guess:8.4f} "
            f"beta_mult={beta_mult:8.4f} "
            f"IE_mult={ie_mult:8.4f} "
            f"beta0={beta0:8.6f} "
            f"IE0={ie0:8.6f}"
        )
nv = len(variant_labels)

# ============================================================
# BUILD PHYLOGENY CROSS IMMUNITY MATRIX
# ============================================================

if use_phylogeny:

    cross_immunity_matrix = build_cross_immunity_matrix(
        variant_labels,
        family_tree
    )

else:

    nv = len(variant_labels)

    cross_immunity_matrix = [
        [0.50 for _ in range(nv)]
        for _ in range(nv)
    ]

    for i in range(nv):
        cross_immunity_matrix[i][i] = 0.90

cross_immunity_flat = []

for row in cross_immunity_matrix:
    cross_immunity_flat.extend(row)

print()
print("Phylogeny cross immunity matrix built")
print()

for i in range(min(5, nv)):
    for j in range(min(5, nv)):
        print(
            f"{variant_labels[i]:12s} -> "
            f"{variant_labels[j]:12s} = "
            f"{cross_immunity_matrix[i][j]:.2f}"
        )

print()
print(f"nv = {len(variant_labels)}")
print()
cross_immunity_text = ", ".join(
    f"{x:.2f}" for x in cross_immunity_flat
)

#
# Future age-aware adjustment
#

# if mean_age_days < 30:
#
#     age_factor = mean_age_days / 30.0
#
#     ie_cap = 0.40 + 0.60 * age_factor
#
#     ie_mult = min(ie_mult, ie_cap)

# ============================================================
# EDIT NAMELIST
# ============================================================

with open(nml_path, "r") as f:
    lines = f.readlines()

new_lines = []

skip_cross_continuation = False

for line in lines:

    stripped = line.strip()

    if skip_cross_continuation:
        if "=" not in stripped and not stripped.startswith("/"):
            continue
        skip_cross_continuation = False

    if stripped.startswith("netcdf_filename"):
        new_lines.append(
            f"  netcdf_filename = 'maps_output_{member_name}.nc',\n"
        )
        continue

    if stripped.startswith("init_i_file"):
        new_lines.append(
            f"  init_i_file = '{ic_filename}',\n"
        )
        continue

    if stripped.startswith("output_dir"):
        new_lines.append(
            f"  output_dir = '{outdir}',\n"
        )
        continue

    if stripped.startswith("expname"):
        new_lines.append(
            f"  expname = '{expname}',\n"
        )
        continue

    if stripped.startswith("month"):
        new_lines.append(f"  month = {month},\n")
        continue

    if stripped.startswith("day"):
        new_lines.append(f"  day = {day},\n")
        continue

    if stripped.startswith("year"):
        new_lines.append(f"  year = {year},\n")
        continue

    if stripped.startswith("nv"):
        new_lines.append(f"  nv = {nv},\n")
        continue

    if stripped.startswith("variant_labels"):
        labels = ", ".join([f"'{v}'" for v in variant_labels])
        new_lines.append(f"  variant_labels = {labels},\n")
        continue

    if stripped.startswith("beta0_by_variant"):
        vals = ", ".join([f"{v:.6f}" for v in beta0_values])
        new_lines.append(f"  beta0_by_variant = {vals},\n")
        continue

    if stripped.startswith("IE0_by_variant"):
        vals = ", ".join([f"{v:.6f}" for v in ie0_values])
        new_lines.append(f"  IE0_by_variant = {vals},\n")
        continue

    if stripped.startswith("cross_immunity_reinfection_flat"):
        new_lines.append(
            "  cross_immunity_reinfection_flat = "
            + cross_immunity_text
            + ",\n"
        )
        skip_cross_continuation = True
        continue

    new_lines.append(line)

with open(nml_path, "w") as f:
    f.writelines(new_lines)
    
print()
print(f"Updated namelist: {nml_path}")
print(f"Active variant file: {variant_file}")
print(f"nv = {nv}")
