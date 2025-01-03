import os

def list_files(data_path):
    """List all files in a directory"""
    return os.listdir(data_path)

def list_files_with_extension(data_path, extension):
    """List all files in a directory with a specific extension"""
    return [f for f in os.listdir(data_path) if f.endswith(extension)]

def list_files_with_extension_directory(data_path, extension):
    """List all directories for files in a directory with a specific extension"""
    return [data_path + f for f in os.listdir(data_path) if f.endswith(extension)]
# "FileName": os.path.basename(pdf) # get the file name without the path

def load_text(file_path):
    """Load text from a file"""
    with open(file_path, 'r') as file:
        text = file.read()
        text = str(text)
    return text