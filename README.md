# Progetto Autenticazione Biometrica Per La Sicurezza Dei Sistemi Informatici

L'obiettivo del progetto è quello di sviluppare un sistema di riconoscimento multi-istanza
automatico basato sulle impronte digitali, fornendo una stima quantitativa sulle performance di
accuratezza.

## Requisiti di Sistema

- Interprete Python versione 3.13

## Struttura del progetto

Il progetto si compone secondo la seguente struttura:

- `config.py`: parametri di configurazione inerenti all'estrazione delle features
- `fingerprint.py`: algoritmi e strutture utilizzati per estrarre le features
- `enrollment.py`: estrazione delle features di tutte le impronte contenute dentro
    la cartella `datasets/` al fine di generare il database che verrà immagazzinato nella
    cartella `database/`
- `datasets/`: datasets utilizzati (FVC2002)
    le etichette delle impronte sono organizzate nel seguente modo:

    - `database full tag`: `datasets/FVC2002/db1_11/101_1`
    - `full tag`: `FVC2002/db1_1/101_1`
    - `database tag`: `FVC2002/db1_1/`
    - `finger tag`: `101`
    - `acquisition tag`: `1`
    - `full finger tag`: `FVC2002/db1_1/101`
    - `fingerprint tag`: `101_1`

- `database/`: databse in cui verranno immagazzinate le feature estratte dai datasets
- `performance_evaluation.py`: script che effettua la valutazione delle performance inerenti
    alle metriche di **FAR** ed **FRR**
- `verification.py`: modulo di verifica, che riceve un'identità dichiarata, e un'identita da
    verificare, per verificare se l'impronta dichiarata è genuina o in impostore
- `identification.py`: modulo di identifica, che riceve un'identità dichiarata, per identificarla
    restituendo l'identità che rappresenta
- `references/`: cartella contenente tutt i riferimenti usati per l'estrazione delle features

## Utilizzo

- **enrollment.py**:

    ```shell
    python3.13 enrollment.py
    ```

- **performance_evaluation.py**:

    ```shell
    python3.13 performance_evaluation.py
    ```

- **verification.py**:
    - aiuto sulla modalità d'utilizzo:

        ```shell
        python3.13 verification.py --help
        ```

    - verifica impronte:

        ```shell
        python3.13 verification.py FVC2002/db1_1/101_1 FVC2002/db1_1/101
        ```

- **identification.py**:
    - aiuto sulla modalità d'utilizzo:

        ```shell
        python3.13 identification.py --help
        ```

    - identificazione impronte:

        ```shell
        python3.13 identification.py FVC2002/db1_1/101_1
        ```

## Visualizzazione del progresso di estrazione e matching delle features

La chiavi di configurazione `SHOW_FEATURES_EXTRACTION_PROGRESS` (bool) e
`SHOW_FINGERPRINT_MATCHING_PROGRESS` (bool) fanno in modo che vengano mostrati visivamente i
risultati di ogni passaggio durante il processo di estrazione e matching delle features.

Verranno mostrate delle finestre che possono essere chiuse simultaneamente premendo il tasto
`ESC` e singolarmente (nel caso di finestre multiple) premendo qualsiasi tasto diverso da `ESC`
