#!/bin/bash

MAPS_FORECASTS/scripts/run_forecast.sh \
    05-17-2026 HZ1 1.00 0.50 8 0-7 "" hz_variant_overrides.csv OVERRIDE

MAPS_FORECASTS/scripts/run_forecast.sh \
    05-17-2026 HZ2 1.00 0.50 8 0-7 "" hz_variant_overrides.csv OVERRIDE
