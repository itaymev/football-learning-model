from meta.data_tool import DataReader

class Times():
    def __init__(self) -> None:
        data = DataReader()
        self.times = data._unique_vals_col("Time")
    
    def _get(self) -> list[str]:
        return self.times