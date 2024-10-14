import numpy as np

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from meta.referees import Referees
from meta.teams import Teams
from meta.divs import Divs
from meta.season import Seasons
from meta.dates import Dates
from meta.times import Times

SEASONS = Seasons()._get()
DIVS = Divs()._get()
DATES = Dates()._get()
TIMES = Times()._get()
REFEREES = Referees()._get()
TEAMS = Teams()._get()

class FootballWizard(nn.Module):
    def __init__(self, input_dim):
        super(FootballWizard, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Assuming 3 classes for classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load the model
def load_model(model_path, input_dim):
    model = FootballWizard(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict a single match
def predict_single_match(model, match_features):
    match_tensor = torch.tensor(match_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(match_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Function to prepare input features
def prepare_input_features(home_team, away_team, additional_info=None):
    # Define default or average values for the numerical features
    default_values = {
        'Season': '22-23',
        'Div': 'Premier League',
        'Date': '05/08/2022',
        'Time': '15:00',
        'FTHG': 1.0,  # Full Time Home Goals
        'FTAG': 1.0,  # Full Time Away Goals
        'HTHG': 0.5,  # Half Time Home Goals
        'HTAG': 0.5,  # Half Time Away Goals
        'HTR': 'A',   # Half Time Result, default to 'A' for Away
        'Referee': 'A Taylor',  # Default referee
        'HS': 10.0,   # Home Shots
        'AS': 10.0,   # Away Shots
        'HST': 3.0,   # Home Shots on Target
        'AST': 3.0,   # Away Shots on Target
        'HC': 5.0,    # Home Corners
        'AC': 5.0,    # Away Corners
        'HF': 10.0,   # Home Fouls
        'AF': 10.0,   # Away Fouls
        'HY': 1.0,    # Home Yellow Cards
        'AY': 1.0,    # Away Yellow Cards
        'HR': 0.0,    # Home Red Cards
        'AR': 0.0     # Away Red Cards
    }

    # Update default values with additional information if provided
    if additional_info:
        default_values.update(additional_info)

    # Create the input feature vector
    raw_input_feature = [
        default_values['Season'], # Season
        default_values['Div'],  # Div
        default_values['Date'],  # Date
        default_values['Time'],  # Time
        home_team,              # HomeTeam
        away_team,              # AwayTeam
        default_values['FTHG'], # FTHG
        default_values['FTAG'], # FTAG
        default_values['HTHG'], # HTHG
        default_values['HTAG'], # HTAG
        default_values['HTR'],  # HTR
        default_values['Referee'], # Referee
        default_values['HS'],   # HS
        default_values['AS'],   # AS
        default_values['HST'],  # HST
        default_values['AST'],  # AST
        default_values['HC'],   # HC
        default_values['AC'],   # AC
        default_values['HF'],   # HF
        default_values['AF'],   # AF
        default_values['HY'],   # HY
        default_values['AY'],   # AY
        default_values['HR'],   # HR
        default_values['AR'],   # AR
    ]

    # Encode categorical features
    label_encoders = {
        'Season': LabelEncoder(),
        'Div': LabelEncoder(),
        'Date': LabelEncoder(),
        'Time': LabelEncoder(),
        'HomeTeam': LabelEncoder(),
        'AwayTeam': LabelEncoder(),
        'HTR': LabelEncoder(),
        'Referee': LabelEncoder()
    }

    # Fit label encoders on the unique values of each categorical feature
    label_encoders['Season'].fit(SEASONS)
    label_encoders['Div'].fit(DIVS)
    label_encoders['Date'].fit(DATES)
    label_encoders['Time'].fit(TIMES)
    label_encoders['HomeTeam'].fit(TEAMS) 
    label_encoders['AwayTeam'].fit(TEAMS) 
    label_encoders['HTR'].fit(['H', 'D', 'A']) # Alter results for tournament football
    label_encoders['Referee'].fit(REFEREES)

    # Transform categorical features
    encoded_input_feature = [
        label_encoders['Season'].transform([raw_input_feature[0]])[0],
        label_encoders['Div'].transform([raw_input_feature[1]])[0],
        label_encoders['Date'].transform([raw_input_feature[2]])[0],
        label_encoders['Time'].transform([raw_input_feature[3]])[0],
        label_encoders['HomeTeam'].transform([raw_input_feature[4]])[0],
        label_encoders['AwayTeam'].transform([raw_input_feature[5]])[0],
        raw_input_feature[6],  # FTHG
        raw_input_feature[7],  # FTAG
        raw_input_feature[8],  # HTHG
        raw_input_feature[9],  # HTAG
        label_encoders['HTR'].transform([raw_input_feature[10]])[0],
        label_encoders['Referee'].transform([raw_input_feature[11]])[0],
        raw_input_feature[12],  # HS
        raw_input_feature[13], # AS
        raw_input_feature[14], # HST
        raw_input_feature[15], # AST
        raw_input_feature[16], # HC
        raw_input_feature[17], # AC
        raw_input_feature[18], # HF
        raw_input_feature[19], # AF
        raw_input_feature[20], # HY
        raw_input_feature[21], # AY
        raw_input_feature[22], # HR
        raw_input_feature[23], # AR
    ]

    return encoded_input_feature

def sensitivity_analysis(model, baseline_features, feature_index, variation_range):
    results = []
    for variation in variation_range:
        varied_features = baseline_features.copy()
        varied_features[feature_index] = variation
        prediction = predict_single_match(model, varied_features)
        results.append((variation, prediction))
    return results

if __name__ == "__main__":
    input_dim = 24  # Corrected number of input features
    model_path = 'football-learning-model/folds/model_fold_3.pth'  # Replace with the path to your saved model

    # Load the model
    model = load_model(model_path, input_dim)

    # Prepare input features with only home and away team names and skewed value
    home_team = 'Arsenal'
    away_team = 'Chelsea'
    additional_info = {}  # Inflate the Home Shots value to test the model
    input_features = prepare_input_features(home_team, away_team, additional_info)

    # Define the range of variations for the feature to analyze
    feature_index = 11  # Index of the feature to vary (e.g., Home Shots)
    variation_range = np.linspace(0, 10, 11)  # Vary the feature from 0 to 10

    # Perform sensitivity analysis
    results = sensitivity_analysis(model, input_features, feature_index, variation_range)
    print(results)

    # Predict the result of the match
    prediction = predict_single_match(model, input_features)

    # Mapping of predicted class to result
    result_mapping = {
        0: 'Home Win',
        1: 'Draw',
        2: 'Away Win'
    }

    # Print the predicted result in the desired format
    if prediction == 0:
        print(f"{home_team} Win at Home")
    elif prediction == 1:
        print(f"{home_team} and {away_team} Draw at {home_team}")
    else:
        print(f"{away_team} Win Away")