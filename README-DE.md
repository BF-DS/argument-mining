# Der Einfluss von Prompt Engineering auf große Sprachmodelle im Argument Mining

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![tiktoken](https://img.shields.io/badge/tiktoken-000000?logo=openai&logoColor=white)](https://github.com/openai/tiktoken)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![seaborn](https://img.shields.io/badge/seaborn-4C8CBF?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![graphviz](https://img.shields.io/badge/graphviz-14A0C4?logo=graphviz&logoColor=white)](https://graphviz.gitlab.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Überblick

Dieses Repository enthält Code, Daten und Dokumentation zur Forschung über den Einfluss von Prompt Engineering auf große Sprachmodelle im Bereich Argument Mining.

---

## Inhaltsverzeichnis

- [Repository-Struktur](#repository-struktur)
- [Installation & Einrichtung](#installation--einrichtung)
- [Verwendung des LLM](#verwendung-des-llm)
- [Kontakt](#kontakt)

---

## Repository-Struktur

- `batch_api/` – Daten für die OpenAI Batch API
  - `input/` – Eingabedaten
  - `output/` – Ausgabedaten
- `data/` – Datensätze
  - `original/` – Argument-Annotated-Essays Datensatz (Version 2) von Stab und Gurevych (2017a), heruntergeladen [hier](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)
  - `transformed/` – Verarbeitete Daten
- `graphs/` – Visualisierungen und zugehörige Notebooks
- `prompts/` – Verwendete Prompts
  - `building-blocks/` – Prompt-Bausteine
  - `final-prompts/` – Finale Prompt-Varianten
- `report/` – Datenbasierter Bericht
- `src/` – Python-Module und Hilfsfunktionen
- `1.EDA.ipynb` – Explorative Datenanalyse
- `2.data-transformation.ipynb` – Datenverarbeitung
- `3.llm.ipynb` – Prompt-Erstellung & LLM-Abfragen
- `4.evaluation.ipynb` – Auswertung der LLM-Ergebnisse
- `requirements.txt` – Abhängigkeiten

---

## Installation & Einrichtung

Es wird empfohlen, eine virtuelle Umgebung (z. B. Anaconda) zu verwenden. Installiere die Abhängigkeiten wie folgt:

```bash
conda create -n argument-mining python=3.12 -y
conda activate argument-mining
pip install -r requirements.txt
```

---

## Verwendung des LLM

Um das GPT-4o mini Modell über die OpenAI Batch API zu nutzen, wird ein OpenAI API-Schlüssel benötigt. Dieser sollte in einer `.env`-Datei als `OPENAI_API_KEY` gespeichert werden. Weitere Informationen zur API und zum Schlüssel findest du [hier](https://platform.openai.com/).

---

## Kontakt

Bei Fragen oder Feedback:
- E-Mail: datadrv.ai@gmail.com

---
