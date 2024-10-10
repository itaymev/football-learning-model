from meta.data_tool import DataReader

class Dates():
    def __init__(self) -> None:
        data = DataReader()
        self.date = data._unique_vals_col("Date")
    
    def _get(self) -> list[str]:
        return self.date