from meta.data_tool import DataReader

class Divs():
    def __init__(self) -> None:
        data = DataReader()
        self.divs = data._unique_vals_col("Div")
    
    def _get(self) -> list[str]:
        return self.divs