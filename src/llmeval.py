import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import seaborn as sns
import matplotlib.pyplot as plt

def replace_ids_with_sentences(data):
    """
    Ersetzt die IDs in den argumentativen Beziehungen durch die entsprechenden Texte aus den Major Claims, Claims und Premises. 
    Zur Anwendung auf einen Pandas Dataframe.
    Args:
        data (dict): Dictionary aus den JSON-Dateien mit den Argumentationskomponenten und -beziehungen
    Returns:
        list: Liste von Dictionaries mit den Argumentativen Beziehungen, wobei die IDs durch die entsprechenden Texte ersetzt wurden
    """
    # Create dictionaries to map IDs to sentences
    major_claims = {mc['ID']: mc['Text'] for mc in data['MajorClaims']}
    claims = {c['ID']: c['Text'] for c in data['Claims']}
    premises = {p['ID']: p['Text'] for p in data['Premises']}

    transformed_relations = []
    # Aus den Beziehungen die Origin und Target IDs entnehmen 
    for relation in data['ArgumentativeRelations']:
        origin_id = relation['Origin']
        target_id = relation['Target']
        # Wenn die ID in den Major Claims (Dict) enthalten ist, dann den Text ausgeben
        if origin_id in major_claims:
            origin_text = major_claims[origin_id]
        # Wenn die ID in den Claims (Dict) enthalten ist, dann den Text ausgeben
        elif origin_id in claims:
            origin_text = claims[origin_id]
        # Wenn die ID in den Premises (Dict) enthalten ist, dann den Text ausgeben
        elif origin_id in premises:
            origin_text = premises[origin_id]
        # Wenn die ID nicht in den Major Claims, Claims oder Premises enthalten ist, dann die ID ausgeben
        else:
            origin_text = origin_id

        
        # Für die Grundwahrheiten, da bei den Beziehungen nicht spezifiziert wurde, auf welche MajorClaim sie sich beziehen
        # Dann wird einfach der Text für den ersten MajorClaim herangezogen.
        if target_id.startswith('MC'):
            target_text = major_claims.get(target_id, target_id)
        
        #elif target_id in major_claims:
        #    target_text = major_claims[target_id]
        elif target_id in claims:
            target_text = claims[target_id]
        elif target_id in premises:
            target_text = premises[target_id]
        else:
            target_text = target_id # Notwendig, da bei Beziehungen zu MajorClaims nicht definiert wird um welchen MajorClaim es sich handelt. 

        # Die Beziehungen mit den durch Texten ersetzten IDs in ein neues Dictionary schreiben
        transformed_relations.append({
            'Origin': origin_text, 
            'Relation': relation['Relation'].lower(), # Beziehungen in Kleinbuchstaben umwandeln
            'Target': target_text
        })

    return transformed_relations


def extract_relations(list_of_dicts):
    """
    Extrahiert die Argumentativen Beziehungen aus einer Liste von Dictionaries und gibt sie als Liste von Tupeln zurück.
    """
    tupel = [(rel['Origin'], rel['Relation'], rel['Target']) for rel in list_of_dicts]
    return tupel


def clean_json_string(json_string):
    """
    Probiert den JSON-String in ein JSON-Objekt umzuwandeln. Sollte der JSON-String nicht korrekt bzw. vollständig sein, wird dieser bereinigt. 
    Ein JSON-String ist bspw. nicht korrekt, wenn die geschlossene Klammern fehlen. In diesem Fall wird der JSON-String bis zum letzten gültigen JSON-Objekt extrahiert.
    Das JSON-Objekt wird dann vervollständigt, indem die fehlenden Klammern hinzugefügt werden.
    Beispiel:

    JSON-String: 
    "{"ArgumentativeComponents": [{"ID": "1", "Text": "Major Claim 1"}, {"ID": "2","
    
    Bereinigter JSON-String:
    "{"ArgumentativeComponents": [{"ID": "1", "Text": "Major Claim 1"}]}"

    Args:
        json_string (str): JSON-String
    Returns:
        dict: JSON-Objekt
    """
    try:
        # Probiert den JSON-String in ein JSON-Objekt umzuwandeln
        json_obj = json.loads(json_string)
        return json_obj
    except json.JSONDecodeError as e: # Wenn JSON-String nicht korrekt ist (z.B. fehlende geschlossene Klammer)
        error_pos = e.pos # Position des Fehlers
        # Finde die Position des letzten gültigen JSON-Objekts. Annahme: Das letzte gültige JSON-Objekt endet mit einer geschlossenen eckigen Klammer
        last_valid_json_obj_pos = json_string.rfind("}", 0, error_pos) # rfind(value, start, end) gibt die Position des letzten Vorkommens eines Substrings zurück
        # Extrahieren des letzten gültigen JSON-Objekts anhand der Position
        last_valid_json_obj = json_string[:last_valid_json_obj_pos + 1]
        # Hinzufügen der fehlenden geschlossenen eckigen Klammer und geschlossenen geschweiften Klammer, um das JSON-Objekt zu vervollständigen
        cleaned_json_string = last_valid_json_obj + "]}"
        # Bereinigten String in JSON-Objekt umwandeln
        cleaned_json = json.loads(cleaned_json_string)
        return cleaned_json

def transform_content_to_args(df: pd.DataFrame, col_name: str):
    """
    Extrahiert aus den JSON-Objekten die Argumentkomponenten und -beziehungen. Bei den Beziehungen werden die IDs durch die entsprechenden Texte ersetzt. 
    Die Argumentkomponenten und -beziehungen werden als neue Spalten im Dataframe hinzugefügt. Die urprüngliche Spalte mit den JSON-Objekten wird gelöscht.
    
    Args:
        df (pd.DataFrame): Dataframe mit den JSON-Objekten
        col_name (str): Name der Spalte mit den JSON-Objekten

    Returns:
        pd.DataFrame: Dataframe mit den extrahierten Argumentkomponenten und -beziehungen
    """  
    df[col_name] = df[col_name].apply(clean_json_string) # Bereinigt die JSON-Strings bei Fehlern
    df['relations'] = df[col_name].apply(replace_ids_with_sentences) # Ersetzt die IDs durch die entsprechenden Texte
    df['relations'] = df['relations'].apply(extract_relations) # Extrahiert die Beziehungen als Liste von Tupeln
    # # Extrahiert die Sätze von Major Claims, Claims und Premises aus dem JSON Objekt und fügt sie als neue Spalten hinzu
    df['majorclaims'] = df[col_name].apply(lambda x: [mc['Text'] for mc in x['MajorClaims']])
    df['claims'] = df[col_name].apply(lambda x: [c['Text'] for c in x['Claims']])
    df['premises'] = df[col_name].apply(lambda x: [p['Text'] for p in x['Premises']])
    df = df.drop(columns=[col_name]) # Löscht die ursprüngliche Spalte mit dem JSON Objekt
    return df    

def extract_response_info(response):
    """Extrahiert Informationen aus der API-Antwort und gibt sie als Pandas Series zurück.
    Args:
        response (dict): API-Antwort
    Returns:
        pd.Series: Pandas Series mit den extrahierten Informationen
    """
    response_json = response
    model = response_json['body']['model']
    system_fingerprint = response_json['body']['system_fingerprint']
    content = response_json['body']['choices'][0]['message']['content']
    prompt_tokens = response_json['body']['usage']['prompt_tokens']
    completion_tokens = response_json['body']['usage']['completion_tokens']
    total_tokens = response_json['body']['usage']['total_tokens']
    promot_token_details = response_json['body']['usage']['prompt_tokens_details']
    completion_token_details = response_json['body']['usage']['completion_tokens_details']
    
    
    return pd.Series([model, system_fingerprint, content, prompt_tokens, completion_tokens, total_tokens, promot_token_details, completion_token_details])


def calculate_bleu_score(groundtruth, prediction):
    """
    Berechnet den BLEU Score für die Vorhersage im Vergleich zur Grundwahrheit.
    
    Args:
        groundtruth (str): Text der Grundwahrheit.
        prediction (str): Text der Vorhersage.
    
    Returns:
        score (float): BLEU Score.
    """
    # mit split() Sätze in Wörter aufteilen
    reference_word_list = groundtruth.split()
    candidate_word_list = prediction.split()
    # Glättungsfunktion
    chencherry = SmoothingFunction()
    # Berechnung BLEU score
    score = sentence_bleu([reference_word_list], candidate_word_list, smoothing_function=chencherry.method1)

    # Codebausteine zur Implementierung des BLEU Scores entnommen aus: https://www.nltk.org/api/nltk.translate.bleu_score.html
    
    return score


def calculate_confusion_matrix(actual_list, pred_list, threshold=0.75):
    """
    Berechnung der Teile der Konfusionsmatrix (TP, FN, FP, TN) basierend auf der Ähnlichkeit der Texteabschnitte. 
    Als Ähnlichkeitsmaß wird der BLEU-Score verwendet. 
    Jeder Textabschnitt kann nur einmal zugeordnet werden. 
    Args:
        actual_list: Liste der Texte aus der Grundwahrheit
        pred_list: Liste der Texte aus der Vorhersage
        threshold: Schwellenwert für die Ähnlichkeit (BLEU-Score), um einen Textabschnitt als True Positive zu werten. 
    Returns:
    - tp: True Positives
    - fn: False Negatives
    - fp: False Positives
    - tn: True Negatives
    - similarities: Liste der Ähnlichkeiten der Texteabschnitte
    
    Beschreibung:
    - True Positives (TP): Wenn der vorhergesagte Text mit dem tatsächlichen Text übereinstimmt und die Ähnlichkeit größer oder gleich dem Schwellenwert ist.
    - False Negatives (FN): Sofern ein Text aus der Grundwahrheit nicht zuvor als TP gewertet wurde, wird er als FN gewertet.
    - False Positives (FP): Sofern ein Text aus der Vorhersage nicht zuvor als TP gewertet wurde, wird er als FP gewertet.
    - True Negatives (TN): Da das LLM nur die Argumentationskomponenten vorhersagt und keine nicht-argumentativen Texte, wird TN immer 0 sein.
    """
    tp = fn = fp = tn = 0 # Initialisierung des Counters
    similarities = [] # liste zum speichern der Ähnlichkeiten
    
    # Listen in Tupel umwandeln, um die Verwendung der Texte zu verfolgen (Text, Flag für Verwendung)
    actual_set = {(text, False) for text in actual_list}
    pred_set = {(text, False) for text in pred_list}
    
    # Vergleich der vorhergesagten Texte mit den tatsächlichen Texten 
    for actual_text, actual_used in actual_set: # Iteration über actual_set, bestehend aus dem Tupel actual_text und actual_used
        if actual_used:   # Wenn actual_used True ist beteutet es, dass der Text (actual_text) bereits verwendet wurde. 
            continue    # In diesem Fall wird der Text übersprungen, da bereits zugeordnet.
        max_similarity = 0 # Initialisierung der maximalen Ähnlichkeit. Wird verwemdet, um herauszufinden, welcher Text aus dem pred_set am besten passt.
        best_match = None # Initialisierung des besten Übereinstimmung
        for pred_text, pred_used in pred_set: # Iteration über pred_set, bestehend aus dem Tupel pred_text und pred_used, um den Text mit der höchsten Ähnlichkeit zu finden
            if pred_used:     # Wenn pred_used True ist beteutet es, dass der Text (pred_text) bereits verwendet wurde.
                continue    # In diesem Fall wird der Text übersprungen, da bereits zugeordnet.
            similarity = round(calculate_bleu_score(actual_text, pred_text), 4) # Berechnung der BLEU-Ähnlichkeit vom vorhergesagten Text zum tatsächlichen Text
            #similarity = are_texts_same(actual_text, pred_text) # Berechnung der Ähnlichkeit vom vorhergesagten Text zum tatsächlichen Text
            #similarity = jaccard_similarity(set(actual_text.split()), set(pred_text.split())) # Berechnung der Jaccard-Ähnlichkeit vom vorhergesagten Text zum tatsächlichen Text
            if similarity > max_similarity: # Wenn die Ähnlichkeit größer als die bisherige maximale Ähnlichkeit ist:
                max_similarity = similarity # Update der maximalen Ähnlichkeit
                best_match = (pred_text, pred_used) # Update der besten Übereinstimmung
        similarities.append(max_similarity) # Hinzufügen der maximalen Ähnlichkeit zur Liste
        
        if max_similarity >= threshold: # Sofern die maximale Ähnlichkeit größer oder gleich dem Schwellenwert ist:
            tp += 1  # wird der Text mit der höchsten Ähnlichkeit als True Positive gewertet
            pred_set.remove(best_match) # Update der Flag für Verwendung, indem der Text aus pred_set entfernt wird und
            pred_set.add((best_match[0], True))  # neu hinzugefügt wird, jedoch mit der Flag True, um anzuzeigen, dass er bereits verwendet wurde
            actual_set.remove((actual_text, False)) # wie zuvor für pred_set
            actual_set.add((actual_text, True)) 
  
    # Betrachtung der verbleibenden Texte aus der Grundwahrheit
    for actual_text, actual_used in actual_set:
        if not actual_used: # Sofern der Text nicht bereits als TP gewertet wurde, wird er als FN gewertet
            fn += 1

    # Betrachtung der verbleibenden Texte aus der Vorhersage
    for pred_text, pred_used in pred_set:
        if not pred_used: # Sofern der Text nicht bereits als TP gewertet wurde, wird er als FP gewertet
            fp += 1
    
    # TN kann es in diesem Fall nicht geben, da das LLM nur die Argumentationskomponenten vorhersagt und keine nicht-argumentativen Texte. 
    # Der Wert für TN wird somit bei 0 bleiben.

    return tp, fn, fp, tn, similarities


def get_confusion_matrix(row, ground_truth_col: list, prediction_col: list):
    """
    Berechnung der Teile der Konfusionsmatrix (TP, FN, FP, TN) anhand der Ähnlichkeit der Textabschnitte für die Argumentationskomponenten Major Claims, Claims und Premises.
    Als Ähnlichkeitsmaß wird der BLEU-Score verwendet.

    Args:
        row: Zeile des Dataframes.
        ground_truth_col: Liste der Spaltennamen für die Grundwahrheit. Reihenfolge: Major Claims, Claims, Premises.
        prediction_col: Liste der Spaltennamen für die Vorhersage. Reihenfolge: Major Claims, Claims, Premises.

    Returns:
        pd.Series: Series mit den Werten für TP, FN, FP, TN und Ähnlichkeit für Major Claims, Claims und Premises
    """    
    # Berechnung der Konfusionsmatrix für das erste Paar von Spalten (Major Claims)
    major_claims_tp, major_claims_fn, major_claims_fp, major_claims_tn, major_claims_sim = calculate_confusion_matrix(row[ground_truth_col[0]], row[prediction_col[0]])
    # Berechnung der Konfusionsmatrix für das zweite Paar von Spalten (Claims)
    claims_tp, claims_fn, claims_fp, claims_tn, claims_sim = calculate_confusion_matrix(row[ground_truth_col[1]], row[prediction_col[1]])
    # Berechnung der Konfusionsmatrix für das dritte Paar von Spalten (Premises)
    premises_tp, premises_fn, premises_fp, premises_tn, premises_sim = calculate_confusion_matrix(row[ground_truth_col[2]], row[prediction_col[2]])
          
    # Rückgabe der Werte als Series
    return pd.Series({
        'MajorClaims_TP': major_claims_tp,
        'MajorClaims_FN': major_claims_fn,
        'MajorClaims_FP': major_claims_fp,
        'MajorClaims_TN': major_claims_tn,
        'MajorClaims_Similarity': major_claims_sim,
        'Claims_TP': claims_tp,
        'Claims_FN': claims_fn,
        'Claims_FP': claims_fp,
        'Claims_TN': claims_tn,
        'Claims_Similarity': claims_sim,
        'Premises_TP': premises_tp,
        'Premises_FN': premises_fn,
        'Premises_FP': premises_fp,
        'Premises_TN': premises_tn,
        'Premises_Similarity': premises_sim,
    })


def calculate_relations_confusion_matrix(actual_list, pred_list, threshold=0.75):
    """
    Berechnung von TP, FN, FP und TN für die Argumentationsbeziehungen basierend auf der Ähnlichkeit der Texteabschnitte. Als Ähnlichkeitsmaß wird der BLEU-Score verwendet.
    Ist vom Prinzip wie die Funktion calculate_confusion_matrix aufgebaut, jedoch mit der Besonderheit, dass die Argumentationsbeziehungen aus Tupeln bestehen.
    Args:
        actual_list: Liste der Beziehungen aus der Grundwahrheit.
        pred_list: Liste der vorhergesagten Beziehungen.
        threshold: Schwellenwert für die Ähnlichkeit (BLEU-Score) der Texteabschnitte aus den Beziehungen, um als True Positive gewertet zu werden.
    Returns:
    - tp: True Positives
    - fn: False Negatives
    - fp: False Positives
    - tn: True Negatives
    - similarities: Liste der Ähnlichkeiten
    """
    tp = fn = fp = tn = 0 # Initialisierung des Counters
    similarities = [] # Liste zum Speichern der Ähnlichkeiten

    # Ergänzung der Beziehungen um den Flag für Verwendung
    actual_set = {(origin, stance, target, False) for origin, stance, target in actual_list} 
    pred_set = {(origin, stance, target, False) for origin, stance, target in pred_list}

    for actual_origin, actual_stance, actual_target, actual_used in actual_set:
        if actual_used: # sofern das Beziehungs-Tupel für die tatsächlichen Beziehungen bereits verwendet wurde, wird es übersprungen
            continue
        max_similarity = (0, 0) # Initialisierung der maximalen Ähnlichkeit als Tupel (origin_similarity, target_similarity)
        best_match = None # Initialisierung der besten Übereinstimmung
        for pred_origin, pred_stance, pred_target, pred_used in pred_set:
            if pred_used: # sofern das Beziehungs-Tupel für die vorhergesagten Beziehungen bereits verwendet wurde, wird es übersprungen
                continue
            similarity_origin = round(calculate_bleu_score(actual_origin, pred_origin), 4) # Berechnung der BLEU-Ähnlichkeit für origin und target
            similarity_target = round(calculate_bleu_score(actual_target, pred_target), 4)
            similarity = (similarity_origin, similarity_target) # Ähnlichkeit als Tupel (origin_similarity, target_similarity)
            if similarity > max_similarity: # Wenn die Ähnlichkeit größer als die bisherige maximale Ähnlichkeit ist:
                max_similarity = similarity
                best_match = (pred_origin, pred_stance, pred_target, pred_used)
        similarities.append(max_similarity)
        
        if max_similarity[0] >= threshold and max_similarity[1] >= threshold: # Beide Ähnlichkeiten müssen größer oder gleich dem Schwellenwert sein
            tp += 1 # Wenn das zutrifft, wird der True Positive Counter erhöht
            pred_set.remove(best_match) # Update der Flag für die Verwendung
            pred_set.add((best_match[0], best_match[1], best_match[2], True)) 
            actual_set.remove((actual_origin, actual_stance, actual_target, False)) 
            actual_set.add((actual_origin, actual_stance, actual_target, True))
    
    # Sofern das Beziehungs-Tupel für die tatsächlichen Beziehungen nicht als TP gewertet wurde, wird es als FN gewertet
    for actual_origin, actual_stance, actual_target, actual_used in actual_set:
        if not actual_used: 
            fn += 1

    # Sofern das Beziehungs-Tupel für die vorhergesagten Beziehungen nicht als TP gewertet wurde, wird es als FP gewertet
    for pred_origin, pred_stance, pred_target, pred_used in pred_set:
        if not pred_used: 
            fp += 1

    # Da das LLM keine nicht-argumentativen Texte vorhersagt, wird TN immer 0 sein.
    tn = 0

    return tp, fn, fp, tn, similarities

def get_relations_confusion_matrix(row, ground_truth_col: str, prediction_col: str):
    """
    Berechnung der Teile der Konfusionsmatrix (TP, FN, FP, TN) anhand der Ähnlichkeit der Textabschnitte für die Argumentationskomponenten Major Claims, Claims und Premises.
    Als Ähnlichkeitsmaß wird der BLEU-Score verwendet.
    Args:
        row: Zeile des Dataframes.
        ground_truth_col: Spaltenname für die Grundwahrheit der Beziehungen.
        prediction_col: Spaltenname für die Vorhersage der Beziehungen.
    Returns:
        pd.Series: Series mit den Werten für TP, FN, FP, TN und Ähnlichkeit für die Beziehungen
    """
    tp, fn, fp, tn, similarities = calculate_relations_confusion_matrix(row[ground_truth_col], row[prediction_col])
    return pd.Series({
        'Relations_TP': tp,
        'Relations_FN': fn,
        'Relations_FP': fp,
        'Relations_TN': tn,
        'Relations_Similarity': similarities
    })


def plot_confusion_matrices(dataframe):
    """
    Erstellt für jede Reihe in einem DataFrame jeweils eine Konfusionsmatrizen für die Argumentkomponenten und die argumentativen Beziehungen.
    Args:
        dataframe (pd.DataFrame): DataFrame mit den Werten der Konfusionsmatrix 
    """
    for index, row in dataframe.iterrows():
        total_tp_mc = row['MajorClaims_TP']
        total_fn_mc = row['MajorClaims_FN']
        total_fp_mc = row['MajorClaims_FP']
        total_tn_mc = row['MajorClaims_TN']

        total_tp_c = row['Claims_TP']
        total_fn_c = row['Claims_FN']
        total_fp_c = row['Claims_FP']
        total_tn_c = row['Claims_TN']

        total_tp_p = row['Premises_TP']
        total_fn_p = row['Premises_FN']
        total_fp_p = row['Premises_FP']
        total_tn_p = row['Premises_TN']

        total_tp_r = row['Relations_TP']
        total_fn_r = row['Relations_FN']
        total_fp_r = row['Relations_FP']
        total_tn_r = row['Relations_TN']

        # Konfusionsmatrizen erstellen
        confusion_matrix_mc = [[total_tp_mc, total_fn_mc], [total_fp_mc, total_tn_mc]]
        confusion_matrix_c = [[total_tp_c, total_fn_c], [total_fp_c, total_tn_c]]
        confusion_matrix_p = [[total_tp_p, total_fn_p], [total_fp_p, total_tn_p]]
        confusion_matrix_r = [[total_tp_r, total_fn_r], [total_fp_r, total_tn_r]]

        # Plotten der Konfusionsmatrizen
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        sns.heatmap(confusion_matrix_mc, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'{index} - Major Claims')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['True', 'False'])
        axes[0, 0].set_yticklabels(['True', 'False'])

        sns.heatmap(confusion_matrix_c, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'{index} - Claims')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_xticklabels(['True', 'False'])
        axes[0, 1].set_yticklabels(['True', 'False'])

        sns.heatmap(confusion_matrix_p, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'{index} - Premises')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_xticklabels(['True', 'False'])
        axes[1, 0].set_yticklabels(['True', 'False'])

        sns.heatmap(confusion_matrix_r, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'{index} - Relations')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xticklabels(['True', 'False'])
        axes[1, 1].set_yticklabels(['True', 'False'])

        plt.tight_layout()
        plt.show()


def calc_mertics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnung der Metriken Precision, Recall und F1-Score für die Argumentkomponenten und Beziehungen.
    Args:
        dataframe (pd.DataFrame): DataFrame mit den Werten der Konfusionsmatrix (TP, FN, FP, TN) für die Argumentkomponenten und Beziehungen.
    Returns:
        pd.DataFrame: DataFrame mit den berechneten Metriken Precision, Recall und F1-Score für die Argumentkomponenten und Beziehungen.

    """
    eval_metrics = []

    for index, row in dataframe.iterrows():
        total_tp_mc = row['MajorClaims_TP']
        total_fn_mc = row['MajorClaims_FN']
        total_fp_mc = row['MajorClaims_FP']
        
        total_tp_c = row['Claims_TP']
        total_fn_c = row['Claims_FN']
        total_fp_c = row['Claims_FP']
        
        total_tp_p = row['Premises_TP']
        total_fn_p = row['Premises_FN']
        total_fp_p = row['Premises_FP']
        
        total_tp_r = row['Relations_TP']
        total_fn_r = row['Relations_FN']
        total_fp_r = row['Relations_FP']
        
        # Berechnung der Metriken für Hauptaussagen
        precision_mc = total_tp_mc / (total_tp_mc + total_fp_mc) if (total_tp_mc + total_fp_mc) != 0 else 0
        recall_mc = total_tp_mc / (total_tp_mc + total_fn_mc) if (total_tp_mc + total_fn_mc) != 0 else 0
        f1_score_mc = 2 * (precision_mc * recall_mc) / (precision_mc + recall_mc) if (precision_mc + recall_mc) != 0 else 0

        # Berechnung der Metriken für Behauptungen
        precision_c = total_tp_c / (total_tp_c + total_fp_c) if (total_tp_c + total_fp_c) != 0 else 0
        recall_c = total_tp_c / (total_tp_c + total_fn_c) if (total_tp_c + total_fn_c) != 0 else 0
        f1_score_c = 2 * (precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) != 0 else 0
        
        # Berechnung der Metriken für Prämissen
        precision_p = total_tp_p / (total_tp_p + total_fp_p) if (total_tp_p + total_fp_p) != 0 else 0
        recall_p = total_tp_p / (total_tp_p + total_fn_p) if (total_tp_p + total_fn_p) != 0 else 0
        f1_score_p = 2 * (precision_p * recall_p) / (precision_p + recall_p) if (precision_p + recall_p) != 0 else 0

        # Berechnung der Metriken für Beziehungen
        precision_r = total_tp_r / (total_tp_r + total_fp_r) if (total_tp_r + total_fp_r) != 0 else 0
        recall_r = total_tp_r / (total_tp_r + total_fn_r) if (total_tp_r + total_fn_r) != 0 else 0
        f1_score_r = 2 * (precision_r * recall_r) / (precision_r + recall_r) if (precision_r + recall_r) != 0 else 0

        # Hinzufügen der berechneten Metriken zu einer Liste von Dictionaries, um daraus einen DataFrame zu erstellen
        eval_metrics.append({
            'Prompt': row['prompt'],
            'Precision_MC': round(precision_mc, 2),
            'Recall_MC': round(recall_mc, 2),
            'F1_Score_MC': round(f1_score_mc, 2),
            'Precision_C': round(precision_c, 2),
            'Recall_C': round(recall_c, 2),
            'F1_Score_C': round(f1_score_c, 2),
            'Precision_P': round(precision_p, 2),
            'Recall_P': round(recall_p, 2),
            'F1_Score_P': round(f1_score_p, 2),
            'Precision_R': round(precision_r, 2),
            'Recall_R': round(recall_r, 2),
            'F1_Score_R': round(f1_score_r, 2),
        })
    
    df = pd.DataFrame(eval_metrics)

    return df


def filter_and_sample(df, component, threshold=0.75, sample_size=5, random_state=42):
    """
    Filtert die Zeilen aus einem DataFrame unterhalb des Ähnlichkeitswerts für eine Argumentationskomponente und gibt eine zufällige Stichprobe zurück.
    Die Stichprobe besteht aus Listen mit Texten aus der Grundwahrheit, der LLM-Ausgabe und den Ähnlichkeitswerten. Kann zur Betrachtung von Fehlern in der LLM-Ausgabe verwendet werden.
    Args:
        df (pd.DataFrame): DataFrame mit den Daten zur Grundwahrheit, LLM-Ausgabe und Ähnlichkeitswerten
        component (str): Name der Argumentationskomponente (MajorClaims, Claims, Premises)
        threshold (float): Schwellenwert für die Ähnlichkeitsmetrik. Texte mit einem Ähnlichkeitswert unterhalb dieses Schwellenwerts werden ausgewählt.
        sample_size (int): Anzahl der zufälligen Stichprobe
        random_state (int): Seed für die Reproduzierbarkeit der Stichprobe
    Returns:
        pd.DataFrame: Zufällige Stichprobe von Texten aus der Grundwahrheit, LLM-Ausgabe und Ähnlichkeitswerten
    """
    similarity_col = f'{component}_Similarity'
    truth_col = f'{component.lower()}_truth'
    llm_col = f'{component.lower()}_llm'

    # Filtern der Zeilen basierend auf dem Schwellenwert 
    row_seperator = df[df[similarity_col].apply(lambda x: min(x) < threshold)]
    col_seperator = [truth_col, llm_col, similarity_col]
    filtered_df = df.loc[row_seperator.index, col_seperator]
    print(f"Anzahl an betroffenen Zeilen: {filtered_df.shape[0]}\n")

    sample = filtered_df.sample(sample_size, random_state=random_state)
    sample_truth = sample[truth_col].values
    sample_llm = sample[llm_col].values
    sample_similarity = sample[similarity_col].values

    for i in range(len(sample_truth)):
        print(f"Grundwahrheit: {sample_truth[i]}")
        print(f"LLM-Ausgabe: {sample_llm[i]}")
        print(f"Änhlichkeitswerte: {sample_similarity[i]}")
        print("\n")

    return sample