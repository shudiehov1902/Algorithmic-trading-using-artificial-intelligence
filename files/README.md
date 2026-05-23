# Implementácia bakalárskej práce

Táto zložka obsahuje zdrojový kód použitý v experimentálnej časti bakalárskej práce.
Kód pripravuje dátovú množinu, trénuje porovnávané modely a obsahuje aj automatizované testy vybraných pomocných výpočtov.

## Cieľ práce

Cieľom práce je overiť, či má pri algoritmoch pre obchodovanie zmysel trénovať
modely nielen na presnú predikciu budúceho denného výnosu, ale aj priamo na
metriky bližšie obchodnému rozhodovaniu. Preto sa porovnávajú modely učené
pomocou MSE a MAE s modelmi, ktorých cieľ je odvodený od Sharpeho a Sortinovho
pomeru.

V experimentoch sa porovnávajú tri rodiny modelov: MLP, LSTM a StockMixer.
Model pre každú akciu vytvorí skóre a z týchto skóre sa následne skladá jednoduché
long-only top-K portfólio. Výsledky sa hodnotia po započítaní transakčných nákladov,
najmä pomocou net Sharpe, net Sortino, Alpha IR, kumulatívneho výnosu a priemerného
obratu portfólia.

## Predpoklady

Na spustenie projektu je potrebné mať:

- Python 3.10 alebo novší,
- nainštalované knižnice z `requirements.txt`,
- pre testy aj knižnice z `requirements-dev.txt`,
- denné OHLC CSV dáta pre jednotlivé tickery,
- odporúčané: GPU s podporou CUDA pre rýchlejší tréning modelov.

Odovzdaná verzia bola kontrolovaná s týmito verziami hlavných nástrojov:

- Python 3.13.2,
- PyTorch 2.6.0,
- NumPy 2.2.3,
- pandas 2.2.3,
- scikit-learn 1.6.1,
- SciPy 1.15.2,
- Matplotlib 3.10.1,
- pytest 8.4.2.

## Inštalácia závislostí

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
```

## Príprava dát

Predvolená cesta k dátam použitá na serveri bola:

```text
/data/alpaca/alpaca_sp500_etf_2025_1day_open_filled
```

Dáta sa pripravia príkazom:

```bash
python3 prepare_dataset.py
```

Ak sú zdrojové CSV súbory inde, cestu možno zadať explicitne:

```bash
python3 prepare_dataset.py --data_dir /cesta/k/csv/suborom --out_dir data
```

## Spustenie experimentov

Všetky modely sa spustia príkazom:

```bash
./run_all_models.sh
```

Spustenie od konkrétneho modelu:

```bash
./run_all_models.sh stock_mixer_sharpe
```

Logy sa ukladajú do adresára `logs`.

## Testy

Automatizované testy sa spustia príkazom:

```bash
python3 -m pytest -q
```

Testy kontrolujú najmä prípravu dát, samplovanie podľa dní, výpočet portfóliových metrík, transakčné náklady a základnú spustiteľnosť skriptov cez `--help`.

## Grafy výsledkov

Grafy použité v práci sa generujú príkazom:

```bash
python3 make_result_plots.py
```

Skript vytvára PDF grafy pre porovnanie hlavných metrík a vzťah medzi turnoverom a čistým Sharpeho pomerom.
Výstupné obrázky sa ukladajú do adresára `img/`.
