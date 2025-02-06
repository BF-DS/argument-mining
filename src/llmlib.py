import tiktoken

#TODO Funktion ggf. auslagern, da sie in EDA und main genutzt wird
# Count tokens by counting the length of the list returned by .encode().
def num_tokens_from_string(string: str, model_name: str) -> int:
    """
    Gibt die Anzahl der Tokens zurück, die ein String für den Tokenzier eines bestimmten Modells hat.
    Args:
        string (str): Der zu tokenisierende String
        model_name (str): Der Name des Modells, für das die Tokenisierung erfolgen soll
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens