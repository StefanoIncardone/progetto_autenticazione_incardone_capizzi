# Progetto Autenticazione Biometrica Per La Sicurezza Dei Sistemi Informatici

L'obiettivo del progetto è quello di sviluppare un sistema di riconoscimento multi-istanza
automatico basato sulle impronte digitali, fornendo una stima quantitativa sulle performance di
accuratezza.

## Requisiti di Sistema

- Interprete Python versione 3.13

## Struttura del progetto

Il progetto si compone secondo la seguente struttura:

### Estrazione delle features

- [`config.py`](config.py): parametri di configurazione inerenti all'estrazione delle features
- `fingerprint.py`: algoritmi e strutture utilizzati per estrarre le features
- `references/`: cartella contenente tutt i riferimenti usati per l'estrazione delle features

### Creazione database

- `enrollment.py`: estrazione delle features di tutte le impronte contenute dentro
    la cartella `datasets/` al fine di generare il database che verrà immagazzinato nella
    cartella `database/`
- `datasets/`: datasets utilizzati (FVC2006)
    le etichette delle impronte sono organizzate nel seguente modo:

    - `database full tag`: `datasets/FVC2006/db1_11/101_1`
    - `full tag`: `FVC2006/db1_1/101_1`
    - `database tag`: `FVC2006/db1_1/`
    - `finger tag`: `101`
    - `acquisition tag`: `1`
    - `full finger tag`: `FVC2006/db1_1/101`
    - `fingerprint tag`: `101_1`
- `database/`: databse in cui verranno immagazzinate le feature estratte dai datasets

### Verifica e Identificazione

- `verification.py`: modulo di verifica, che riceve un'identità dichiarata, e un'identita da
    verificare, per verificare se l'impronta dichiarata è genuina o in impostore
- `identification.py`: modulo di identifica, che riceve un'identità dichiarata, per identificarla
    restituendo l'identità che rappresenta

### Valutazione delle performance

- `performance_evaluation.py`: script che effettua la valutazione delle performance inerenti
    alle metriche di **FAR** ed **FRR**

### Visualizzazione del progresso di estrazione e matching delle features

- `features_extraction.py`:

## Utilizzo

- **enrollment.py**:

    ```shell
    python3.13 enrollment.py
    ```

- **verification.py**:
    - aiuto sulla modalità d'utilizzo:

        ```shell
        python3.13 verification.py --help
        ```

    - verifica impronte:

        ```shell
        python3.13 verification.py FVC2006/db1_1/101_1 FVC2006/db1_1/101
        ```

- **identification.py**:
    - aiuto sulla modalità d'utilizzo:

        ```shell
        python3.13 identification.py --help
        ```

    - identificazione impronte:

        ```shell
        python3.13 identification.py FVC2006/db1_1/101_1
        ```

- **performance_evaluation.py**:

    ```shell
    python3.13 performance_evaluation.py
    ```
