# Masterarbeit - Der Einfluss von Prompt Engineering auf Large Language Models im Argument Mining
Dieses Repository enthält die Abschlussarbeit von Benjamin Frank Fels für den Masterstudiengang Data Science & Management (M.Sc.) an der Digital Business University of Applied Sciences (DBU) zur Erlangung des Grades Master of Science.

## Aufbau des Repositories
Das Repository ist in folgende Ordner unterteilt. Die Jupyter-Notebooks 1-4 enthalten den Code zur Durchführung der Arbeit. Die Notebooks sind in der Reihenfolge ihrer Nummerierung auszuführen.
- `batch_api/`: Enthält die Daten für die OpenAI Batch-API
  - `input/`: Eingabedaten für die Batch-API
  - `output/`: Ausgabedaten der Batch-API
- `data/`: Enthält den originalen Datensatz und die verarbeiteten Daten
  - `original/`: Originaler Datensatz
  - `transformed/`: Verarbeitete Daten
- `graphs/`: Visualisierungen inklusive dazugehörigem Jupyter-Notebook
- `prompts/`: Verwendete Prompts
  - `building-blocks/`: Prompt-Bausteine für die Erstellung der Prompts
  - `final-prompts/`: Prompt-Variationen zur Anwendung auf das Large Language Model
- `src/`: Enthält Python-Dateien mit eigenen Funktionen zum Importieren in die Jupyter-Notebooks
- `1.EDA.ipynb`: Jupyter-Notebook zur explorativen Datenanalyse
- `2.data-transformation.ipynb`: Jupyter-Notebook zur Transformation der ann-Dateien in JSON-Dateien
- `3.llm.ipynb`: Jupyter-Notebook zur Erstellung der Prompts und Anfragen an das Large Language Model
- `4.evaluation.ipynb`: Jupyter-Notebook zur Evaluation der Ausgaben des Large Language Models
- `report/`: Enthält den datenbasierten Report zur Masterarbeit
- `requirements.txt`: Abhängigkeiten für das Projekt

## Installation der Abhängigkeiten
Zur Ausführung der Jupyter-Notebooks empfiehlt es sich die Abhängigkeiten in einer eigenen Umgebung wie beispielsweise mittels Anaconda zu installieren. Die Abhängigkeiten können über die Datei requirements.txt installiert werden. Sofern Anaconda installiert ist, kann die Umgebung mit den folgenden Befehlen im Terminal aufgesetzt und aktiviert werden.

```bash
$ conda create -n masterarbeit_bfels python=3.12.6 -y

$ conda activate masterarbeit_bfels

$ pip install -r requirements.txt
```

## Verwendung des LLMs
Das Large Language Model GPT-4o mini wird über die OpenAI Batch-API verwendet. Hierzu muss ein API-Key von OpenAI angefordert werden. Über den nachfolgenden Link gelangt man zu einer Anleitung zur Anforderung des API-Keys: https://www.geeksforgeeks.org/how-to-get-your-own-openai-api-key/#how-to-obtain-your-openai-api-key 

In diesem Projekt ist es vorgesehen, dass der API-Key in einer .env-Datei unter der Variable `OPENAI_API_KEY` gespeichert wird. Diese Datei wird in den Jupyter-Notebooks geladen. Nach der Installation der Abhängigkeiten und dem Speichern des API-Keys in der .env-Datei kann der Code aus den Jupyter-Notebooks ausgeführt werden.

## Hinweis zur Sprache
Die Kommentare und Beschriftungen sind in Deutsch zum besseren Verständnis der Arbeit. Der Code ist hingegen in Englisch, da die Bibliotheken und Dokumentationen meist in Englisch sind. 

## Kontakt
Bei Fragen wenden Sie sich bitte an:
- benjamin.fels@student.dbuas.de