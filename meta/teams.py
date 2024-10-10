from meta.data_tool import DataReader

class Teams():
    def __init__(self) -> None:
        data = DataReader()
        self.teams = data._unique_vals_col("HomeTeam")
    
    def _get(self) -> list[str]:
        return self.teams