import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.features = None
        self.target = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_le = LabelEncoder()

    def load_data(self):
        self.data = pd.read_csv(self.file_path, low_memory=False)
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'].fillna(pd.to_datetime(self.data['Date'], format='%d/%m/%y', errors='coerce', dayfirst=True)))

    def clean_data(self):
        char_columns = ['Season', 'Div', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee', 'Time', 'Date']
        num_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

        for column in char_columns:
            self.data[column] = self.data[column].astype(str)

        for column in num_columns:
            self.data[column] = self.data[column].astype(float)

        self.data[char_columns] = self.data[char_columns].fillna('Not Available')
        self.data[num_columns] = self.data[num_columns].fillna(0.0)

    def preprocess_data(self):
        self.features = self.data.drop(columns=['FTR'])
        self.target = self.data['FTR']

        print("Columns before encoding and scaling:", self.features.columns)

        for column in self.features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.features[column] = le.fit_transform(self.features[column])
            self.label_encoders[column] = le

        self.features = self.scaler.fit_transform(self.features)
        self.target = self.target_le.fit_transform(self.target)

        print("Number of columns after encoding and scaling:", self.features.shape[1])

    def get_preprocessed_data(self):
        if self.data is None:
            self.load_data()
        self.clean_data()
        self.preprocess_data()
        return self.features, self.target

    def check_for_nan(self):
        if pd.isnull(self.features).any().any():
            print("NaN values found in features:")
            print(pd.isnull(self.features).sum())
            raise ValueError("Data contains NaN values in features. Please clean the data.")

        if pd.isnull(self.target).any():
            print("NaN values found in target:")
            print(pd.isnull(self.target).sum())
            raise ValueError("Data contains NaN values in target. Please clean the data.")
        
def main():
    file_path = 'Project 1/football-data-top-5-european-leagues/past-data.csv'  # Replace with the path to your CSV file
    preprocessor = Preprocessor(file_path)

    try:
        # Load, clean, and preprocess the data
        preprocessor.load_data()
        preprocessor.clean_data()
        preprocessor.preprocess_data()

        # Check for NaN values
        preprocessor.check_for_nan()

        # Get the preprocessed data
        features, target = preprocessor.get_preprocessed_data()

        # Print the shapes of the features and target to verify
        print("Features shape:", features.shape)
        print("Target shape:", target.shape)

        # Verify the number of attributes
        if features.shape[1] == 24:
            print("Preprocessing successful: 24 attributes present.")
        else:
            print(f"Preprocessing error: {features.shape[1]} attributes present instead of 24.")

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()