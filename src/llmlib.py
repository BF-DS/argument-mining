import tiktoken # open-source Tokenizer von OpenAI
import pandas as pd
import json
import os

def num_tokens_from_string(string: str, model_name: str) -> int:
    """
    Gibt die Anzahl der Tokens zurück, die ein String für den Tokenzier eines bestimmten Modells hat. Als Tokenizer wird tiktoken verwendet.
    Args:
        string (str): Der zu tokenisierende String
        model_name (str): Der Name des Modells, für das die Tokenisierung erfolgen soll
    Returns:
        int: Die Anzahl der Tokens, die der String für das Modell hat
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def generate_batch_input(test_df: pd.DataFrame, prompt_df: pd.DataFrame, file_name: str, schema: json, path="batch_api/input/"):
    """
    Funktion zur Erstellung von gesammelten Anfragen an das LLM in einem Format, sodass sie über die Batch-API gestellt werden können. Die Anfragen werden in einer JSONL-Datei gespeichert.
    Args:
        test_df: DataFrame mit Testdaten
        prompt_df: DataFrame mit Prompts
        file_name: Name der JSONL-Datei
        schema: JSON-Schema als strukturiertes Ausgabeformat für das LLM.
    Returns:
        jsonl_output: JSONL-Datei mit den gesammelten Anfragen für die Batch-API
    """
    temperature = 0
    llm_seed = 123
    model = "gpt-4o-mini"

    dict_list = []
    # iteration über Zero-Shot-Prompts
    for _, prompt_row in prompt_df.iterrows():
        # Iteration über Testdaten
        for _, test_df_row in test_df.iterrows():
            custom_id_str = prompt_row['prompt_name'] + "_" + test_df_row['txt_file']# + "_" + str(id_counter)
            # write batch input for jsonl file
            input_dict = {"custom_id": custom_id_str, 
                          "method": "POST", "url": "/v1/chat/completions",
                          "body": {"model": model,
                                   "messages": [{"role": "developer", "content": prompt_row['prompt_txt']}, # system Rolle wurde in developer umbenannt
                                                {"role": "user", "content": "Text: " + test_df_row['txt']}], # user Rolle für Eingaben des Nutzers wie bei ChatGPT 
                                                "temperature": temperature,
                                                "seed": llm_seed,
                                                "response_format": {
                                                    "type": "json_schema", # wichtig festzulegen, da sonst Fehlermeldung
                                                    "json_schema": {
                                                        "name": "ArgumentMiningExtraction", # wichtig festzulegen, da sonst Fehlermeldung
                                                        "schema": schema,
                                                        "strict": True 
                                                    }
                                                    }
                                                }
                                     }
            dict_list.append(input_dict)

    jsonl_output = "\n".join(json.dumps(item) for item in dict_list)

    # Output in JSONL-Datei schreiben
    file_path = f"{path}{file_name}.jsonl"
    with open(file_path, 'w') as f:
        f.write(jsonl_output)

    # Quelle zur Verwendung der Batch API: https://platform.openai.com/docs/guides/batch?lang=python
    # Codebausteine zur Textgenerierung entnommen aus: https://platform.openai.com/docs/guides/text-generation
    # Siehe dort auch bei "Messages and roles" für die Rollenbeschreibungen 

    return file_path


def split_jsonl_file(input_file_path, max_tokens=20_000_000):
    """
    Teilt eine einzlne JSONL-Datei in mehrere Dateien auf, anhand der maximalen Anzahl an Tokens pro Datei. 
    Die Dateien werden im gleichen Verzeichnis wie die ursprüngliche Datei gespeichert. Die ursprüngliche Datei wird gelöscht, da sie nicht mehr benötigt wird. 
    Das spart Speicherplatz.
    Args:
        input_file_path: Dateiname inkl. Pfad der zu teilenden JSONL-Datei
        max_tokens: Maximale Anzahl an Tokens pro Datei
    """
    input_path = os.path.dirname(input_file_path) 

    def write_to_file(data, file_index):
        """
        Schreibt die Daten in eine JSONL-Datei.
        Args:
            data: Daten, die in die Datei geschrieben werden sollen
            file_index: Index der Datei
        """
        with open(f"{input_path}/batch_input_{file_index}.jsonl", 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    current_tokens = 0
    file_index = 1
    current_data = []

    with open(input_file_path, 'r') as f:
        # Iteration über die Zeilen der JSONL-Datei
        for line in f:
            json_obj = json.loads(line) # Laden der JSON-Objekte
            messages = json_obj.get("body", {}).get("messages", []) # Extrahieren der Nachrichten aus dem JSON-Objekt
            messages_str = json.dumps(messages) # Umwandlung der Nachrichten in einen String
            tokens = num_tokens_from_string(messages_str, model_name="gpt-4o-mini") # Berechnung der Tokenanzahl für die Nachrichten

            if current_tokens + tokens > max_tokens: # wenn die Anzahl der bisherigen Tokens plus die Tokens des aktuellen Objekts größer als das Limit ist
                write_to_file(current_data, file_index) # schreibe die Daten in die Datei
                file_index += 1 # erhöhe den Dateiindex
                current_data = [] # setze die aktuellen Daten zurück
                current_tokens = 0 # setze die aktuellen Tokens zurück
            
            # wenn die Anzahl der bisherigen Tokens plus die Tokens des aktuellen Objekts kleiner als das Limit ist. Mit einem Else-Statement würde das letzte Objekt nicht in die Datei geschrieben werden.
            current_data.append(json_obj) # füge das aktuelle Objekt zu den aktuellen Daten hinzu
            current_tokens += tokens # erhöhe die Anzahl der aktuellen Tokens um die Tokens des aktuellen Objekts

    if current_data: # wenn es nach der Iteration noch Daten gibt, die noch nicht in eine Datei geschrieben wurden
        write_to_file(current_data, file_index) # schreibe die Daten in die Datei

    # löschen der ursprünglichen Datei
    os.remove(input_file_path)
    
    return None

# Codebausteine zur Erstellung der nachfolgenden Hilfsfunktionen zur Anwendung der Batch-API wurden entnommen aus: https://platform.openai.com/docs/guides/batch?lang=python 
def upload_batch_file(filepath, client):
    """
    Lädt die JSONL-Datei mit den gesammelten Anfragen auf die OpenAI-Plattform hoch.
    Args:
        filepath (str): Pfad zur JSONL-Datei
        client (OpenAI): OpenAI(api_key)
    Returns:
        response (dict): Antwort der Batch-API
    """
    response = client.files.create(
        file=open(filepath, 'rb'),
        purpose='batch'
    )
    return response


def create_batch(input_file_id, metadata_dict, client):
    """
    Erstellt einen Batch anhand der zuvor hochgeladenen JSONL-Datei.
    Args:
        input_file_id (str): ID der hochgeladenen JSONL-Datei
        metadata_dict (dict): Metadaten für den Batch
        client (OpenAI): OpenAI(api_key)
    Returns:
        response (dict): Antwort der Batch-API
    """
    response = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata_dict
    )
    return response


def check_batch_status(batch_id, client):
    """
    Überprüft den Status eines Batches anhand der Batch-ID.
    Args:
        batch_id (str): ID des Batches
        client (OpenAI): OpenAI(api_key)
    Returns:
        response (dict): Antwort der Batch-API
    """
    response = client.batches.retrieve(batch_id)
    print(f"Status: {response.status}")

    if response.status == "failed":
        print(f"Error: {response.errors}")
    elif response.status == "in_progress" or response.status == "validating" or response.status == "finalizing":
        print("Der Batch wird noch verarbeitet. Bitte warten und später erneut prüfen.")

    print(f"\nBeschreibung des Batches: {response.metadata["description"]}")
    print(f"Anfragen gesamt: {response.request_counts.total}")
    print(f"Davon erfolgreich: {response.request_counts.completed}")
    print(f"Davon fehlerhaft: {response.request_counts.failed}")
    if response.output_file_id is not None:
        print(f"Erfolgreiche Abfragen können abgerufen werden mit ID: {response.output_file_id}")
    else:
        print("Keine erfolgreichen Abfragen zum herunterladen vorhanden.")
    
    if response.error_file_id is not None:
        print(f"Für weiter Informationen zum Fehler Abfrage an Error-File mit ID: {response.error_file_id}\n")
    else:
        print("Keine fehlerhaften Abfragen zum herunterladen vorhanden.\n")
    
    return response

def retrieve_and_save_batch_results(batch_file_id, output_path, file_name, client):
    """
    Lädt die Ausgaben eines Batches herunter und speichert diese in einer JSONL-Datei. Das funktioniert nur, wenn der Batch-Status "completed" ist.
    Args:
        batch_file_id (str): ID der Batch-Datei
        file_name (str): Name der Ausgabedatei
        output_path (str): Pfad, in dem die Ausgabedatei gespeichert werden soll
        client (OpenAI): OpenAI(api_key)
    Returns:
        results (str): Ergebnisse des Batches
    """
    file_response = client.files.content(batch_file_id)
    results = file_response.text
    with open(output_path + file_name + ".jsonl", 'w') as f:
        f.write(results)
    return results