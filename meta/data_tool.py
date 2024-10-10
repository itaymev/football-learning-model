import pandas as pd

# A tool for me to learn about the data while making this project

# I also used it for dynamic data upload

# https://www.kaggle.com/datasets/prateekchauhands/football-data-top-5-european-leagues/data
FILE_PATH = 'Project 1/football-data-top-5-european-leagues/past-data.csv'

class DataReader():
    def __init__(self, path=FILE_PATH) -> None:
        self.df = pd.read_csv(path)
        self.len = len(self.df)

    def _headers(self) -> list[str]:
        return self.df.columns.tolist()
    
    def _unique_vals_col(self, col) -> list[str|int|float]:
        # This will be so useful trust
        if col not in self._headers():
            raise ValueError(f"'{col}' does not exist.")

        return self.df[col].unique().tolist()

    def _len(self) -> int:
        # This might seem pointless but O(n) len is still slow so I want to do it just once
        return self.len
 
if __name__ == "__main__":
    data = DataReader()
    print(data._unique_vals_col("Div"))