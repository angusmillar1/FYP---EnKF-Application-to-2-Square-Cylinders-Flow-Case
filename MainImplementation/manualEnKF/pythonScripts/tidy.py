import os
import glob

def delete_Ucollection_files(base_path, pattern):
    """
    Deletes files matching a pattern in a directory.
    
    Parameters:
    - base_path: str
        The base path where the member directories are located.
    - pattern: str
        The pattern of the file to delete inside each member directory.
    """
    # Expand the pattern to match all possible files
    files_to_delete = glob.glob(os.path.join(base_path, "member*/", pattern))

    if not files_to_delete:
        print("No files found matching the pattern.")
        return

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found (already deleted?): {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Define base path and file pattern
base_path = "memberRunFiles"
file_pattern = "U_collection2.txt"

# Run the deletion
delete_Ucollection_files(base_path, file_pattern)
