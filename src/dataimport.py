import os

def list_files(data_path):
    """
    Listet alle Dateien in einem Verzeichnis auf.
    Args:
        data_path: Pfad zum Verzeichnis
    Returns:
        Liste mit Dateinamen
    """
    return os.listdir(data_path)

def list_files_with_extension(data_path, extension):
    """
    Listet alle Dateien in einem Verzeichnis auf, die eine bestimmte Dateiendung haben.
    Args:
        data_path: Pfad zum Verzeichnis
        extension: Dateiendung
    Returns:
        Liste mit Dateinamen
    """
    return [f for f in os.listdir(data_path) if f.endswith(extension)]

def list_files_with_extension_directory(data_path, extension):
    """
    Listet alle Pfade zu Dateien in einem Verzeichnis auf, die eine bestimmte Dateiendung haben.
    Args:
        data_path: Pfad zum Verzeichnis
        extension: Dateiendung
    Returns:
        Liste mit Dateipfaden
    """
    return [data_path + f for f in os.listdir(data_path) if f.endswith(extension)]

def load_text(file_path):
    """
    Lädt den Inhalt einer Textdatei.
    Args:
        file_path: Pfad zur Datei
    Returns:
        Textinhalt der Datei
    """
    with open(file_path, 'r') as file:
        text = file.read()
        text = str(text)
    return text