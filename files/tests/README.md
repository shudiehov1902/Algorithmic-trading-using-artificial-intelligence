# Testy implementácie

Tento adresár obsahuje ľahké testy implementácie spúšťané cez `pytest`.

Súbor testov pokrýva tieto časti:

1. `test_samplers.py`
   - `MultiDateBatchSampler`
2. `test_dataset_builder.py`
   - `load_price_panel`
   - `build_dataset_arrays`
3. `test_portfolio_utils.py`
   - risk metrics
   - robust validation split logic
   - buffer backtest and fee application
4. `test_cli_smoke.py`
   - `--help`
   - import / syntax smoke checks

Cieľom nie je úplné pokrytie celého projektu. Prioritou sú deterministické
kontroly prípravy dát, výpočtu portfólia a validačnej výberovej logiky.
