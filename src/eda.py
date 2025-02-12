import os
import pandas as pd
# eigene Funktion
from src.llmlib import num_tokens_from_string

def calculate_txt_file_lengths(files_path_list, model_id):
    """
    Ermitteltt die Anzahl der Zeichen, Wörter und Tokens für eine Liste von Textdateien.
    Args:
        files_path_list: Liste von Dateipfaden
        model_id: Modell-ID
    Returns:
        DataFrame mit Dateinamen, Textinhalt, Anzahl der Zeichen, Wörter und Tokens
    """
    file_lengths = []

    for file_path in files_path_list:
        with open(file_path, 'r') as f:
            
            content = f.read()
            char_count = len(content) # count characters
            word_count = len(content.split()) # count words
            token_count = num_tokens_from_string(content, model_id) # count tokens
            file_lengths.append({'FileName': os.path.basename(file_path),
                                 'txt': content, 
                                 'CharCount': char_count,
                                 'WordCount': word_count,
                                 'TokenCount_txt': token_count})

    df_lengths = pd.DataFrame(file_lengths)
    return df_lengths


def count_entities(files_list, model_id):
    """
    Zählt die Anzahl der Hauptaussagen, Behauptungen, Prämissen und Beziehungen in einer Liste von .ann-Dateien.
    Args:
        files_list: Liste von Dateipfaden
        model_id: Modell-ID
    Returns:
        DataFrame mit Dateinamen, Anzahl der Hauptaussagen, Behauptungen, Prämissen, Beziehungen und insgesamt annotierten Tokens
    """
    data = []
    for file_path in files_list:
        file_name = os.path.basename(file_path)
        majorclaims = 0
        claims = 0
        premises = 0
        #stances = 0
        relation = 0 # stances werden als Beziehungen gezählt
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('T'):
                    entity = line.split('\t')[1].split(' ')[0]
                    if entity == 'MajorClaim':
                        majorclaims += 1
                    elif entity == 'Claim':
                        claims += 1
                    elif entity == 'Premise':
                        premises += 1
                elif line.startswith('A'):
                    entity = line.split('\t')[1].split(' ')[0]
                    if entity == 'Stance':
                        relation += 1
                elif line.startswith('R'):
                    relation += 1
            token_count = num_tokens_from_string(' '.join(lines), model_id)

        data.append({'FileName': file_name,
                     'MajorClaims': majorclaims,
                     'Claims': claims,
                     'Premises': premises,
                     #'Stances': stances,
                     'Relations': relation,
                     'Total': majorclaims + claims + premises + relation,
                     'TokenCount_ann': token_count})
    
    df = pd.DataFrame(data)
    return df

def load_ann_files(files_path_list):
    """
    Laden der .ann-Dateien in ein Pandas DataFrame mit den Spalten 'FileName' und 'Content'.
    Args:
        files_path_list: Liste von Dateipfaden
    Returns:
        DataFrame mit Dateinamen und Inhalt der .ann-Dateien
    """
    data = []

    for file_path in files_path_list:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r') as file:
            content = file.read()
        data.append({'FileName': file_name, 'Content': content})

    df = pd.DataFrame(data)
    return df