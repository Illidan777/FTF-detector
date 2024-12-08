import os

import joblib


def serialize_objects(file_path, *objects, overwrite=False):
    """
    Serializes one or more objects into a file.

    Parameters:
        file_path (str): The path to the file where the objects will be saved.
        *objects: One or more objects to be serialized.
        overwrite (bool): Whether to overwrite the file if it already exists. Default is False.

    Exceptions:
        - FileExistsError: Raised if the file already exists and overwrite is False.
        - OSError: Raised if there are issues while saving the file.
    """
    try:
        # Check if the file already exists
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"The file {file_path} already exists and overwrite is set to False.")

        # Save the objects to the file using joblib
        joblib.dump(objects, file_path)

        print(f"Objects successfully serialized into {file_path}")

    except FileExistsError as fe:
        print(f"Error: {fe}")
        raise  # Re-raise the exception for further handling if needed
    except OSError as e:
        print(f"Error while writing to the file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def deserialize_objects(file_path):
    """
    Deserializes objects from a file.

    Parameters:
        file_path (str): The path to the file from which the objects will be loaded.

    Returns:
        list: A list of objects loaded from the file.

    Exceptions:
        - FileNotFoundError: Raised if the file is not found.
        - OSError: Raised if there are issues while reading the file.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        # Load the objects from the file using joblib
        objects = joblib.load(file_path)

        print(f"Objects successfully deserialized from {file_path}")
        return objects

    except FileNotFoundError as fnf:
        print(f"Error: {fnf}")
    except OSError as e:
        print(f"Error while reading the file: {e}")
    except joblib.externals.loky.process_executor.TimeoutError as te:
        print(f"Timeout error during deserialization: {te}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
