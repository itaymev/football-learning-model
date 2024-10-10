from meta.data_tool import DataReader

class Referees():
    def __init__(self) -> None:
        data = DataReader()
        self.refs = data._unique_vals_col("Referee")
    
    def _get(self) -> list[str]:
        return self.refs