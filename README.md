# Masterarbeit - Der Einfluss von Prompt Engineering auf Large Language Models im Argument Mining
Dieses Repository enthält die Abschlussarbeit von Benjamin Frank Fels für den Masterstudiengang Data Science & Management (M.Sc.) an der Digital Business University of Applied Sciences (DBU) zur Erlangung des Grades Master of Science.

## Aufbau des Repositories
```
.
├── 1.EDA.ipynb                     # Explorative Datenanalyse
├── 2.data-transformation.ipynb     # Datenumwandlung und -vorbereitung
├── 3.llm-pipeline.ipynb            # Pipeline für Large Language Models
├── 4.evaluation.ipynb              # Evaluierung der Modelle
├── ann-file-formatation.ipynb      # Annotation und Dateiformatierung
├── batch_api/                      # Batch-API für die Verarbeitung
├── claim_data_profiling_report.html# Bericht zur Datenprofilierung
├── data/                           # Rohdaten und Zwischenergebnisse
├── dataframe.csv                   # Konsolidierte Daten im CSV-Format
├── graphs/                         # Grafiken und Visualisierungen
│   └── Prompt-Struktur.pptx        # Präsentation zur Prompt-Struktur
├── prompts/                        # Verwendete Prompts
├── README.md                       # Diese README-Datei
└── report/                         # Abschlussbericht und Dokumentation
```

# Installation 
Zur Ausführung der Jupyter-Notebooks empfiehlt es sich die Abhängigkeiten in einer eigenen Umgebung wie beispielsweise Anaconda zu installieren. Die Abhängigkeiten können über die Datei requirements.txt installiert werden. Sofern Anaconda installiert ist, kann die Umgebung mit den folgenden Befehlen im Terminal aufgesetzt und aktiviert werden.

```bash
$ conda create -n masterarbeit_bfels python=3.12.6 -y

$ conda activate masterarbeit_bfels

$ pip install -r requirements.txt
```

## Verwendung des LLMs
Das Large Language Model (LLM) wird über die Batch-API verwendet. Hierzu muss ein API-Key von OpenAI angefordert  werden.
Link zu einer Anleitung: https://www.geeksforgeeks.org/how-to-get-your-own-openai-api-key/#how-to-obtain-your-openai-api-key 

In diesem Projekt ist es vorgesehen, dass der API-Key in einer .env-Datei unter der Variable `OPENAI_API_KEY`gespeichert wird. Diese Datei wird dann in den Notebooks geladen.


Anschließend sollte der Code in den Jupyter-Notebooks ausführbar sein.

## Hinweis zur Sprache
Die Kommentare und Beschriftungen sind in Deutsch zum besseren Verständnis der Arbeit. Der Code ist hingegen in Englisch, da die Bibliotheken und Dokumentationen meist in Englisch sind. 

## Kontakt
Bei Fragen oder Anmerkungen wenden Sie sich bitte an:
- benjamin.fels@student.dbuas.de