import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Load the dataset from a given filepath."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df

def preprocess_data(df, scaler, model, expected_columns):
    """Process the entire dataset with detailed debugging information."""
    
    # Check initial state of DataFrame
    print("Initial DataFrame columns:")
    print(df.columns.tolist())
    
    # Ensure the 'Rating' column remains the last column
    if 'Rating' in df.columns:
        y = df['Rating']
        df = df.drop('Rating', axis=1)  # Drop 'Rating' column for feature processing
        print("Dropped 'Rating' column for preprocessing.")
    else:
        raise ValueError("The 'Rating' column is missing in the dataset.")
    
    # Identify categorical columns for one-hot encoding
    categorical_columns = [
        'TransportationMode', 'PhysicalActivities', 'DietType', 'LocalFoodFrequency', 
        'EnergySource', 'HomeType', 'ClothingFrequency', 'SustainableBrands', 'CommunityInvolvement', 
        'Gender', 'UsingPlasticProducts', 'DisposalMethods'
    ]

    # Clean categorical columns to ensure consistency (e.g., remove extra spaces, capitalize)
    print("Cleaning categorical columns...")
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.strip().str.capitalize()
        print(f"Cleaned column: {col}")

    # One-hot encode categorical columns (drop_first=False to retain all columns)
    print("Applying one-hot encoding...")
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    print("One-hot encoding complete.")
    
    # Print columns after one-hot encoding for debugging
    print("Columns after one-hot encoding:")
    print(df.columns.tolist())

    # Renaming columns to match the expected format
    print("Renaming specific columns to match the model's format...")
    df.rename(columns={
        'DietType_Mostly animal-based': 'DietType_Mostly Animal-Based',
        'DietType_Mostly plant-based': 'DietType_Mostly Plant-Based',
        'TransportationMode_Public transit': 'TransportationMode_Public Transit',
        'EnvironmentalAwareness': 'EnvironmentalAwareness_2',  # This should be handled based on actual values
    }, inplace=True)

    # Ensure that all required columns are present after encoding
    print("Checking for missing columns...")
    missing_columns = set(expected_columns) - set(df.columns)
    print(f"Missing columns identified: {missing_columns}")

    # Add missing columns with default value 0
    if missing_columns:
        print("Adding missing columns...")
    for col in missing_columns:
        df[col] = 0  # Add missing columns with default 0
        print(f"Added missing column: {col} with default value 0")

    # Handle PhysicalActivities (Ensure all possible categories are present)
    print("Ensuring PhysicalActivities categories are present...")
    physical_activity_columns = ['PhysicalActivities_High', 'PhysicalActivities_Low', 'PhysicalActivities_Moderate', 'PhysicalActivities_Nan']
    for col in physical_activity_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing categories for PhysicalActivities
            print(f"Added missing PhysicalActivities column: {col} with default value 0")
    
    # Reorder columns to match the expected input
    print("Reordering columns to match expected model input...")
    df = df[expected_columns]

    # Print the columns before scaling
    print("Columns before scaling:")
    print(df.columns.tolist())

    # Numerical features to scale
    numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']
    
    # Check if any numerical columns are missing
    print("Checking for missing numerical columns...")
    missing_cols = set(numerical_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    # Scale the numerical features using the scaler loaded
    print("Scaling numerical features...")
    df[numerical_features] = scaler.transform(df[numerical_features])
    print("Scaling complete.")

    # Add the 'Rating' column back to the DataFrame as the last column
    df['Rating'] = y
    print("Added 'Rating' column back to the DataFrame.")

    # Print final columns before saving
    print("Final columns after preprocessing and scaling:")
    print(df.columns.tolist())
    print("Data preprocessing complete.")

    return df

if __name__ == '__main__':
    # Load data
    filepath = 'data/raw/dataset.csv'
    print(f"Starting preprocessing for file: {filepath}")
    df = load_data(filepath)

    # Load model and scaler
    model_path = 'models/random_forest_model.joblib'
    scaler_path = 'models/scaler.joblib'
    print(f"Loading model from {model_path}...")
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")

    # Define expected columns (manually or from a known source)
    expected_columns = [
        'ParticipantID', 'Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption',
        'Location_Suburban', 'Location_Urban', 'Location_Rural', 'DietType_Mostly Animal-Based', 'DietType_Mostly Plant-Based',
        'DietType_Mostly Balanced', 'TransportationMode_Car', 'TransportationMode_Public Transit', 'TransportationMode_Walk',
        'TransportationMode_Bike', 'EnergySource_Non-Renewable', 'EnergySource_Renewable', 'EnergySource_Mixed',
        'HomeType_Apartment', 'HomeType_House', 'HomeType_Other', 'Gender_Male', 'Gender_Female', 'Gender_Non-Binary',
        'Gender_Prefer not to say', 'LocalFoodFrequency_Often', 'LocalFoodFrequency_Always', 'LocalFoodFrequency_Rarely',
        'LocalFoodFrequency_Sometimes', 'ClothingFrequency_Often', 'ClothingFrequency_Always', 'ClothingFrequency_Rarely',
        'ClothingFrequency_Sometimes', 'SustainableBrands_TRUE', 'SustainableBrands_FALSE', 'EnvironmentalAwareness_1',
        'EnvironmentalAwareness_2', 'EnvironmentalAwareness_3', 'EnvironmentalAwareness_4', 'EnvironmentalAwareness_5',
        'CommunityInvolvement_Low', 'CommunityInvolvement_High', 'CommunityInvolvement_Moderate', 'CommunityInvolvement_None',
        'UsingPlasticProducts_Often', 'UsingPlasticProducts_Rarely', 'UsingPlasticProducts_Sometimes', 'UsingPlasticProducts_Never',
        'DisposalMethods_Composting', 'DisposalMethods_Landfill', 'DisposalMethods_Recycling', 'DisposalMethods_Combination',
        'PhysicalActivities_Low', 'PhysicalActivities_Moderate', 'PhysicalActivities_None', 'PhysicalActivities_High'
    ]

    # Preprocess the data
    processed_df = preprocess_data(df, scaler, model, expected_columns)

    # Save the processed data to a new CSV
    output_path = 'data/processed/processed_data.csv'
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete, processed data saved to {output_path}.")
