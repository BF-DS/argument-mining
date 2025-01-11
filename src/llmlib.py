import tiktoken

#TODO Funktion ggf. auslagern, da sie in EDA und main genutzt wird
# Count tokens by counting the length of the list returned by .encode().
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens