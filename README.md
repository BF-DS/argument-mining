# The Impact of Prompt Engineering on Large Language Models in Argument Mining

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

## Overview

This repository contains code, data, and documentation for research on the impact of prompt engineering on large language models in the field of argument mining.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Using the LLM](#using-the-llm)
- [Contact](#contact)

---

## Repository Structure

- `batch_api/` – Data for the OpenAI Batch API
  - `input/` – Input data
  - `output/` – Output data
- `data/` – Datasets
  - `original/` – Argument-Annotated-Essays dataset (Version 2) by Stab and Gurevych (2017a), downloaded [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)
  - `transformed/` – Processed data
- `graphs/` – Visualizations and related notebooks
- `prompts/` – Used prompts
  - `building-blocks/` – Prompt building blocks
  - `final-prompts/` – Final prompt variants
- `report/` – Data-based report
- `src/` – Python modules and helper functions
- `1.EDA.ipynb` – Exploratory data analysis
- `2.data-transformation.ipynb` – Data processing
- `3.llm.ipynb` – Prompt creation & LLM queries
- `4.evaluation.ipynb` – Evaluation of LLM results
- `requirements.txt` – Dependencies

---

## Installation & Setup

It is recommended to use a virtual environment (e.g., Anaconda). Install the dependencies as follows:

```bash
conda create -n argument-mining python=3.12 -y
conda activate argument-mining
pip install -r requirements.txt
```

---

## Using the LLM

To use the GPT-4o mini model via the OpenAI Batch API, an OpenAI API key is required. Store this in a `.env` file as `OPENAI_API_KEY`. More information about the API and key can be found [here](https://platform.openai.com/).

---


## Contact

For questions or feedback:
- Email: datadrv.ai@gmail.com

---