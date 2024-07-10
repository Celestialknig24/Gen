import os
import traceback
import pandas as pd
from langchain.document_loaders import GenericLoader
from langchain.schema import Document

def read_file_content(file_path):
    """
    Attempt to read the content of a file, regardless of its format.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path} as text: {e}")
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return content.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error reading file {file_path} as binary: {e}")
            traceback.print_exc()
            return None

def read_excel_content(file_path):
    """
    Attempt to read the content of an Excel file.
    """
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
        content = ""
        for sheet_name, df in excel_data.items():
            content += f"Sheet: {sheet_name}\n"
            content += df.to_string(index=False)
            content += "\n\n"
        return content
    except Exception as e:
        print(f"Error reading Excel file {file_path}: {e}")
        traceback.print_exc()
        return None

def file_loader(folder_path):
    data = []
    unread_files = []
    
    # Get all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Attempt to load the document using the GenericLoader
                loader = GenericLoader.from_filesystem(
                    path=file_path,
                    glob=file,
                    suffixes=[os.path.splitext(file)[1]],  # Get file extension
                    show_progress=True
                )
                docs = loader.lazy_load()
                
                for doc in docs:
                    if not isinstance(doc, Document):
                        doc = Document(doc)
                    data.append(doc)
                    print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error loading {file} with GenericLoader: {e}")
                traceback.print_exc()
                
                # Fallback: specific handling for Excel files
                if file.endswith('.xlsx'):
                    content = read_excel_content(file_path)
                else:
                    # Fallback: try to read the file content manually
                    content = read_file_content(file_path)
                
                if content:
                    doc = Document(content)
                    data.append(doc)
                    print(f"Successfully processed {file} with fallback method")
                else:
                    unread_files.append(file)
    
    print(f"Processed {len(data)} files, {len(unread_files)} files could not be read")
    return data, unread_files

# Example usage:
folder_path = '/path/to/your/folder'
data, unread_files = file_loader(folder_path)
