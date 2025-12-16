Skelet prezentácie – Obchodovanie s pomocou umelej inteligencie: Sharpe/Sortino ratio ako loss funkcia

Poznámka: Všetky nadpisy a odrážky sú po slovensky, pripravené na kopírovanie do PowerPoint / Google Slides. Štýl: stručné body, jasná hierarchia, „populárny“ tón, jemné emoji pre vizuálny dôraz.

---

Slide 1 – Titulný slide

Nadpis:

> Obchodovanie s pomocou umelej inteligencie: Sharpe/Sortino ratio ako loss funkcia

Podnadpis:

- Bakalárska práca
- Autor: Vlad Shudegov
- Vedúci: [meno vedúceho]
- Fakulta / Katedra: [doplň]
- Akademický rok: [doplň]

---

Slide 2 – Motivácia

Nadpis:

> Motivácia

Body:

- Finančné trhy sú veľmi volatilné a ťažko predvídateľné.
- V praxi sa čoraz viac používajú modely umelej inteligencie na obchodovanie.
- Väčšina modelov optimalizuje štatistickú chybu (MSE, MAE), nie priamo finančný výkon.
- Obchodníkov však zaujíma skôr:
  - dlhodobý výnos,
  - riziko,
  - Sharpeho a Sortinov pomer.

Poznámka pre rečníka: „Chcem skúmať, či sa dá model učiť tak, aby sa zlepšovali finančné metriky, nie len štatistická chyba.“

---

Slide 3 – Cieľ práce

Nadpis:

> Cieľ práce

Body:

- Vybudovať a otestovať modely na predikciu denných výnosov akcií s použitím neurónových sietí.
- Porovnať:
  - rôzne loss funkcie (MSE, MAE),
  - rôzne architektúry (MLP, LSTM).
- Vyhodnocovať modely z pohľadu:
  - štatistických metrík (MSE, MAE),
  - Sharpeho a Sortinovho pomeru obchodnej stratégie.
- Pripraviť pôdu pre ďalší krok – Sharpe/Sortino ako loss funkcia.

---

Slide 4 – Teoretický základ: výnos a riziko

Nadpis:

> Výnos a riziko

Body:

- Denný výnos akcie:
  [ r_t = (close_t - close_{t-1}) / close_{t-1} ]
- Priemerný denný výnos = očakávaný zisk za deň.
- Riziko reprezentujeme pomocou:
  - smerodajnej odchýlky výnosov (volatilita),
  - pre Sortino len záporné výnosy (downside risk).
- V obchodovaní je dôležitý pomer výnos / riziko, nie len samotný výnos.

---

Slide 5 – Sharpeho a Sortinov pomer

Nadpis:

> Sharpeho a Sortinov pomer

Body:

- Sharpeho pomer (zjednodušene, bez bezrizikovej sadzby):
  [ Sharpe = E[r] / sigma(r) ]
- Sortinov pomer:
  [ Sortino = E[r] / sigma(r_{r<0}) ]
- Interpretácia:
  - Sharpe > 0 → stratégia zarába v prepočte na riziko,
  - čím vyšší Sharpe/Sortino, tým lepší „risk-adjusted“ výkon.
- V tejto práci hodnotím stratégie podľa anualizovaného Sharpeho a Sortina.

---

Slide 6 – Problém klasických loss funkcií

Nadpis:

> Prečo nestačí MSE / MAE

Body:

- Bežné modely predikujú hodnotu výnosu a minimalizujú:
  - MSE (Mean Squared Error),
  - MAE (Mean Absolute Error).
- Tieto funkcie merajú:
  - priemernú štatistickú chybu predikcie,
  - neberú do úvahy:
    - riziko stratégie,
    - asymetriu výnosov,
    - transakčné rozhodnutia (long / flat).
- Cieľ ďalšej práce:
  - skúmať loss funkcie, ktoré sú priamo založené na Sharpe/Sortino.

---

Slide 7 – Dáta

Nadpis:

> Dáta

Body:

- Zdroj: server fakulty, priečinok: /data/alpaca/alpaca_sp500_etf_2025_1day_open_filled
- Denné dáta (1-day sviečky) pre:
  - akcie indexu S&P 500,
  - viaceré ETF (napr. SPY, QQQ, sektorové ETF).
- Každý súbor TICKER.csv obsahuje:
  - open_date, open, high, low, close, volume_base.
- Po spracovaní:
  - 542 tickerov,
  - spolu ~1,29 milióna riadkov.

---

Slide 8 – Predspracovanie dát

Nadpis:

> Predspracovanie dát

Body:

- Pre každý ticker zvlášť:
  - výpočet denného výnosu ret z ceny close,
  - vytvorenie 10 lagov výnosov:
    - ret_lag_1 … ret_lag_10.
- Každý riadok reprezentuje:
  - konkrétny deň,
  - konkrétnu akciu,
  - vstup: 10 posledných denných výnosov,
  - cieľ: výnos nasledujúceho dňa.
- Spojenie všetkých tickerov do jedného veľkého DataFrame df_all.
- Rozdelenie podľa dátumu:
  - Train: 2016 – 2022
  - Validation: 2023
  - Test: 2024 – 2025

---

Slide 9 – Škálovanie a finálny dataset

Nadpis:

> Škálovanie a finálny dataset

Body:

- Na vstupné lagované výnosy používam StandardScaler:
  - fit na X_train,
  - transform na X_val a X_test.
- Po štandardizácii (train):
  - priemer ≈ 0,
  - smerodajná odchýlka ≈ 1.
- Finálne rozmery (multi-asset):
  - X_train: (912 838, 10)
  - X_val: (134 206, 10)
  - X_test: (244 294, 10)

---

Slide 10 – Základný model: MLP

Nadpis:

> Model MLP

Body:

- Typ: Multilayer Perceptron (MLP) – dopredná neurónová sieť.
- Vstup: vektor dĺžky 10 (lagované výnosy).
- Architektúra:
  - Linear(10 → 64) + ReLU
  - Linear(64 → 32) + ReLU
  - Linear(32 → 1)
- Výstup: predikovaný denný výnos (\hat{r}_t).
- Optimalizácia:
  - Adam, learning rate = 0,001,
  - loss funkcia: MSE (pri jednom experimente aj MAE).

---

Slide 11 – Experiment 1: jedna akcia (ADBE)

Nadpis:

> Experiment 1: jedna akcia (ADBE)

Body:

- Prvý experiment: model MLP trénovaný len na akcii ADBE (Adobe).
- Dáta:
  - Train/Val/Test rozdelené podľa rokov (2016–2022, 2023, 2024+).
- Výsledky MLP + MSE (približne):
  - Test MSE ≈ 7 × 10^{-4},
  - Test MAE ≈ 1,8 %.
- Obchodná stratégia (long, ak predikcia > 0):
  - priemerný denný výnos: negatívny,
  - Sharpe anualizovaný ≈ -0,83,
  - Sortino anualizovaný ≈ -0,77.
- Buy & Hold na ADBE v rovnakom období:
  - tiež negatívny Sharpe ≈ -0,83.

Záver:

- Na jednej akcii sú výsledky nestabilné a stratégia bola stratová.

---

Slide 12 – Motivácia pre multi-asset prístup

Nadpis:

> Prečo multi-asset dataset

Body:

- Trénovanie na jednej akcii:
  - malý počet dát,
  - model sa prispôsobí špecifikám jedného titulu,
  - zlá generalizácia.
- Riešenie:
  - použiť viac akcií (S&P 500 + ETF),
  - učiť jeden model na veľkom množstve príkladov,
  - získať robustnejšie vzorce správania trhu.

---

Slide 13 – Experiment 2: MLP + MSE na multi-asset

Nadpis:

> Experiment 2: MLP + MSE (multi-asset)

Body:

- Vstup: 10 lagovaných denných výnosov pre akýkoľvek ticker.
- Tréning na celom multi-asset datasete:
  - X_train: 912 838 riadkov
  - X_val: 134 206
  - X_test: 244 294
- Loss funkcia: MSE.
- Výsledné chyby:
  - Train MSE ≈ 4,43 × 10^{-4},
  - Val MSE ≈ 3,62 × 10^{-4},
  - Test MSE ≈ 4,33 × 10^{-4}.

---

Slide 14 – Výsledky: MLP + MSE (multi-asset)

Nadpis:

> Výsledky – MLP + MSE

Body:

- Obchodná stratégia:
  - pravidlo: ak (\hat{r}_t > 0) → long, inak 0,
  - výnos stratégie = (signal_t · r_t).
- Na testovacej množine:
  - priemerný denný výnos ≈ 0,0608 %,
  - denná volatilita ≈ 2,04 %,
  - Sharpe anualizovaný ≈ 0,47,
  - Sortino anualizovaný ≈ 0,60.
- Buy & Hold (na rovnakom období a tituloch):
  - priemerný denný výnos ≈ 0,0625 %,
  - Sharpe anualizovaný ≈ 0,48.

Záver:

- MLP vytvára ziskovú stratégiu s kladným Sharpeho pomerom, ale výkon je veľmi blízky jednoduchej stratégii Buy & Hold.

---

Slide 15 – LSTM: motivácia

Nadpis:

> Prechod na LSTM

Body:

- MLP pracuje s vstupom ako s vektorom, explicitne nevyužíva časový charakter dát.
- Časové rady (výnosy, ceny) majú:
  - autokorelácie,
  - lokálne trendy,
  - zhluky volatility (volatility clustering).
- LSTM (Long Short-Term Memory):
  - rekurentná neurónová sieť,
  - spracúva sekvenciu krok za krokom,
  - má vnútornú „pamäť“ stavu.
- Cieľ: overiť, či LSTM lepšie zachytí časové vzťahy ako MLP.

---

Slide 16 – Model LSTM

Nadpis:

> Architektúra LSTM modelu

Body:

- Vstup pre LSTM:
  - sekvencia dĺžky 10: [r_{t-10}, r_{t-9}, …, r_{t-1}]
  - každý krok má 1 vstup (denný výnos).
- Architektúra:
  - LSTM vrstva:
    - input_size = 1,
    - hidden_size = 32,
    - num_layers = 1,
    - batch_first = True.
  - Posledný skrytý stav → Linear(32 → 1).
- Výstup: predikovaný denný výnos (\hat{r}_t).
- Loss funkcia: MAE.

---

Slide 17 – Experiment 3: LSTM + MAE (multi-asset)

Nadpis:

> Experiment 3: LSTM + MAE

Body:

- Tréning na tom istom multi-asset datasete ako MLP.
- Loss funkcia: MAE.
- Výsledné chyby (Test):
  - Test MAE ≈ 1,35 % (priemerná absolútna denná chyba výnosu).
- Stratégia (predikcia > 0 → long):
  - priemerný denný výnos ≈ 0,0365 %,
  - denná volatilita ≈ 1,40 %,
  - Sharpe anualizovaný ≈ 0,41,
  - Sortino anualizovaný ≈ 0,31.
- Buy & Hold:
  - Sharpe anualizovaný ≈ 0,48.

Záver:

- LSTM poskytuje ziskovú stratégiu, ale z pohľadu Sharpeho pomeru zaostáva za Buy & Hold aj za MLP modelom.

---

Slide 18 – Experiment 4: LSTM + MSE (multi-asset)

Nadpis:

> Experiment 4: LSTM + MSE (multi-asset)

Body:

- Rovnaký LSTM model ako v predchádzajúcom experimente:
  - vstup: sekvencia 10 denných výnosov,
  - LSTM (hidden size = 32, 1 vrstva) + Linear(32 → 1).
- Rozdiel: loss funkcia = MSE namiesto MAE.
- Výsledné chyby:
  - Train MSE ≈ 4,32 × 10⁻⁴
  - Val MSE ≈ 3,58 × 10⁻⁴
  - Test MSE ≈ 4,30 × 10⁻⁴
- Stratégia (predikcia > 0 → long):
  - priemerný denný výnos ≈ 0,031 %
  - denná volatilita ≈ 1,38 %
  - Sharpe (anualizovaný) ≈ 0,36
  - Sortino (anualizovaný) ≈ 0,29
- Buy & Hold:
  - Sharpe (anualizovaný) ≈ 0,48

Poznámka pre rečníka: „LSTM s MSE má najlepšiu MSE spomedzi modelov, ale z pohľadu Sharpeho pomeru je najslabší.“

---

Slide 19 – Porovnanie MLP a LSTM

Nadpis:

> Porovnanie MLP a LSTM

Body:

- MLP + MSE (multi-asset):
  - Test MSE ≈ 4,33 × 10^{-4},
  - Sharpe anualizovaný ≈ 0,47.
- LSTM + MAE (multi-asset):
  - Test MAE ≈ 1,35 %,
  - Sharpe anualizovaný ≈ 0,41.
- Buy & Hold:
  - Sharpe anualizovaný ≈ 0,48.
- Zhrnutie:
  - Obe modely vytvárajú ziskové stratégie (Sharpe > 0),
  - MLP je bližšie k výkonu Buy & Hold, LSTM má nižší Sharpe,
  - Optimalizácia štatistickej chyby (MSE/MAE) negarantuje maximálny Sharpeho pomer.

---

Slide 20 – Zhrnutie doterajších výsledkov

Nadpis:

> Zhrnutie doterajšej práce

Body:

- Pripravil som veľký multi-asset dataset (S&P 500 + ETF, 2016–2025).
- Implementoval som predspracovanie:
  - výpočet denných výnosov,
  - lagovanie (10 dní),
  - štandardizáciu,
  - časové rozdelenie Train/Val/Test.
- Otestoval som viaceré modely:
  - MLP + MSE (jedna akcia aj multi-asset),
  - MLP + MAE (jedna akcia),
  - LSTM + MAE (multi-asset).
- Vyhodnotil som:
  - MSE/MAE,
  - Sharpeho a Sortinov pomer,
  - porovnanie s Buy & Hold.

---

Slide 21 – Plán ďalšej práce

Nadpis:

> Plán ďalšej práce

Body:

- Systematické porovnanie:
  - MLP vs. LSTM na rovnakých loss funkciách (MSE aj MAE),
  - rôzne hyperparametre (počty neurónov, dĺžka sekvencie).
- Skúmanie Sharpe/Sortino-based loss funkcií:
  - model bude optimalizovať priamo Sharpeho alebo Sortinov pomer.
- Zahrnutie transakčných nákladov a realistických obmedzení.
- Finalizácia teoretickej časti:
  - detailný popis metrík (MSE, MAE, Sharpe, Sortino),
  - vysvetlenie použitých modelov (MLP, LSTM).
- Príprava textu bakalárskej práce a obhajoby.

---

