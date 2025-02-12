import re
import json
import pandas as pd
# eigene Module
from src.dataimport import load_text

def update_ann_file(input_text):
    """
    Aktualisiert die IDs in einer .ann-Datei und gibt den aktualisierten Text zurück.
    Args:
        input_text (str): Der Text der .ann-Datei
    Returns:
        str: Der aktualisierte Text der .ann-Datei
    """
    # Muster
    pattern_unit = r"^(T\d+)\s+(MajorClaim|Claim|Premise)" # ^ Anfang der Zeile, T\d+ T gefolgt von Ziffern, \s+ ein oder mehrere Leerzeichen, MajorClaim|Claim|Premise
    pattern_relation = r"^(R\d+)\s+\w+ Arg1:(T\d+) Arg2:(T\d+)" # ^ Anfang der Zeile, R\d+ R gefolgt von Ziffern, \s+ ein oder mehrere Leerzeichen, \w+ ein oder mehrere Buchstaben, Arg1 & Arg2 :T\d+ Arg1:T gefolgt von Ziffern
    pattern_stance = r"^(A\d+)\s+Stance (T\d+)" # ^ Anfang der Zeile, A\d+ A gefolgt von Ziffern, \s+ ein oder mehrere Leerzeichen, Stance, T\d+ T gefolgt von Ziffern

    # Zähler für die Argumentationskomponente für fortlaufende Nummerierung
    counters = {"MajorClaim": 1, "Claim": 1, "Premise": 1}
    
    # Dictionary für die Zuordnung der alten und neuen IDs
    id_mapping = {} # key: alte ID, value: neue ID

    output_lines = [] # Liste zum speichern der neuen Zeilen

    for line in input_text.splitlines():
        # Argumentationseinheiten anhand der Muster suchen
        match_unit = re.match(pattern_unit, line)
        if match_unit:
            old_id, unit_type = match_unit.groups()
            prefix = {"MajorClaim": "MC", "Claim": "C", "Premise": "P"}[unit_type]
            new_id = f"{prefix}{counters[unit_type]}"
            counters[unit_type] += 1
            id_mapping[old_id] = new_id

            # alte ID durch neue ID ersetzen
            line = line.replace(old_id, new_id, 1)

        # relations anhand der Muster suchen
        match_relation = re.match(pattern_relation, line)
        if match_relation:
            old_rel_id, arg1, arg2 = match_relation.groups()
            new_arg1 = id_mapping.get(arg1, arg1)
            new_arg2 = id_mapping.get(arg2, arg2)

            # alte Argument-ID durch neue Argument-ID ersetzen
            line = re.sub(rf"Arg1:{arg1}", f"Arg1:{new_arg1}", line)
            line = re.sub(rf"Arg2:{arg2}", f"Arg2:{new_arg2}", line)

        # stance anhand der Muster suchen
        match_stance = re.match(pattern_stance, line)
        if match_stance:
            old_a_id, target_id = match_stance.groups()
            new_target_id = id_mapping.get(target_id, target_id)

            # alte Target-ID durch neue Target-ID ersetzen
            line = re.sub(rf"Stance {target_id}", f"Stance {new_target_id}", line)

        output_lines.append(line)
    # zusammenfügen der Zeilen
    return "\n".join(output_lines)



def transform_ann_to_json(text):
    """
    Transformiert den Text der ann-Datei mit den angepassten IDs in ein JSON-Objekt.
    Args:
        text (str): Der Text der angepassten .ann-Datei
    Returns:
        str: Das JSON-Objekt als String
    """
    data = {
        "MajorClaims": [],
        "Claims": [],
        "Premises": [],
        "ArgumentativeRelations": []
    }
    # Text in Zeilen aufteilen
    lines = text.strip().split('\n')
    # Iteration über die Zeilen, um die Annotationen zu extrahieren
    for line in lines:
        parts = line.split()
        # prüfen, ob die Zeile einen Hauptaussage enthält
        if parts[0].startswith('MC'):
            data["MajorClaims"].append({"ID": parts[0], "Text": ' '.join(parts[4:])})
        # prüfen, ob die Zeile eine Behauptung enthält
        elif parts[0].startswith('C'):
            data["Claims"].append({"ID": parts[0], "Text": ' '.join(parts[4:])})
        # prüfen, ob die Zeile eine Prämisse enthält
        elif parts[0].startswith('P'):
            data["Premises"].append({"ID": parts[0], "Text": ' '.join(parts[4:])})
        # prüfen, ob die Zeile eine argumentative Beziehung enthält
        elif parts[0].startswith('R'):
            data["ArgumentativeRelations"].append({
                "Origin": parts[2].split(':')[1],
                "Relation": parts[1],
                "Target": parts[3].split(':')[1]
            })
        # prüfen, ob die Zeile eine Haltung enthält, welche ebenfalls als argumentative Beziehung betrachtet wird
        elif parts[0].startswith('A'):
            data["ArgumentativeRelations"].append({
                "Origin": parts[2],
                "Relation": parts[3],
                "Target": "MC" # Es wurde in den Annotationen nicht spezifiziert, welcher MajorClaim gemeint ist. Deshalb wird nicht fortlauend nummeriert.
            })

    return json.dumps(data, indent=2)   # dumps() überführt das Dictionary in ein JSON-Objekt
                                        # indent sorgt für eine Einrückung des JSON-Objekts


def count_relation_types(json_data_dir_list):
    """
    Zählt die Anzahl der Beziehungen pro Typ in den JSON-Dateien.
    Args:
        json_data_dir_list: Liste mit den Dateipfaden der JSON-Dateien
    Returns:
        DataFrame mit der Anzahl der Beziehungen pro Typ
    """
    relation_count = {"Origin": {}, "Target": {}}
    for file in json_data_dir_list:
        input_text = load_text(file)
        data = json.loads(input_text)
        for relation in data["ArgumentativeRelations"]:
            origin = relation["Origin"]
            target = relation["Target"]
            origin_type = origin[0]
            target_type = target[0]
            
            if origin_type not in relation_count["Origin"]:
                relation_count["Origin"][origin_type] = 0
            relation_count["Origin"][origin_type] += 1
            
            if target_type not in relation_count["Target"]:
                relation_count["Target"][target_type] = 0
            relation_count["Target"][target_type] += 1
            
    df = pd.DataFrame(relation_count)
    return df


def count_origin_target_pairs(json_files):
    """
    Zählt die Anzahl der Vorkommen von Origin-Target-Paaren in den JSON-Dateien.
    Args:
        json_files: Liste mit den JSON-Dateipfaden
    Returns:
        DataFrame mit den Origin-Target-Paaren und deren Anzahl
    """
    pair_count = {}
    for file in json_files:
        input_text = load_text(file)
        data = json.loads(input_text)
        for relation in data["ArgumentativeRelations"]:
            pair = (relation["Origin"][0], relation["Target"][0])
            if pair not in pair_count:
                pair_count[pair] = 0
            pair_count[pair] += 1
    df = pd.DataFrame(pair_count.items(), columns=["Origin-Target Pair", "Count"])
    df = df.sort_values(by="Count", ascending=False)
    return df