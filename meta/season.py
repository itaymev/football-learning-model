from meta.data_tool import DataReader

class Seasons():
    def __init__(self) -> None:
        data = DataReader()
        self.seasons = data._unique_vals_col("Season")
    
    def _get(self) -> list[str]:
        return self.seasons