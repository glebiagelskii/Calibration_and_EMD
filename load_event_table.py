"""This is a script for loading event table data.
 Please note that there is a standardised format for event tables available for inspection in standard_formats.txt.

 Non-standard event tables should be used via adapters (their design is in progress)
 """

from pathlib import Path
import pandas as pd
def load_event_table(path_to_file: str) -> pd.DataFrame:
    path = Path(path_to_file)

    if path.suffix == '.csv':
        df = pd.read_csv(path)
        print('Read csv succesfully!')
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
        print('Read parquet succesfully!')
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}. Event table should be either csv or parquet. \n Please use an adapter or a different version for other file extensions.')
    return df



event_table = load_event_table('stuff.csv')



def table_header_format_is_correct(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    standard_columns = set(pd.read_csv('standard_event_table_format.csv').columns)
    if columns == standard_columns:
        return True
    else:
        print(f'Non-standard header format. \n The culprit columns are {columns - standard_columns}')
        return False
table_header_format_is_correct(event_table)