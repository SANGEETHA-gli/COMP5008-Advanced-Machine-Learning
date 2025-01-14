import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


red_wine_path = r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\winequality-red.csv"
white_wine_path = r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\winequality-white.csv"

try:
    red_wine_data = pd.read_csv(red_wine_path, sep=";")
    white_wine_data = pd.read_csv(white_wine_path, sep=";")
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit()


red_wine_data["type"] = "red"
white_wine_data["type"] = "white"


combined_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)


features = combined_data.drop(columns=["quality", "type"])  
target = combined_data[["quality", "type"]]  

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_features_df = pd.DataFrame(
    normalized_features, columns=features.columns
)


preprocessed_data = pd.concat([normalized_features_df, target.reset_index(drop=True)], axis=1)


try:
    preprocessed_path = r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\preprocessed_winequality.csv"
    preprocessed_data.to_csv(preprocessed_path, index=False)
    print(f"Data preprocessing complete. Preprocessed dataset saved as '{preprocessed_path}'.")
except Exception as e:
    print(f"Error saving preprocessed data: {e}")
