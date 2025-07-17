import pandas as pd
import joblib

# Load your model and scaler
model = joblib.load(r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\random_forest_model.joblib')
scaler = joblib.load(r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\scaler.joblib')  # Load the scaler

def map_user_input(user_input):
    # Corrected mapping logic
    transformed_input = {
        'ParticipantID': user_input.get('ParticipantID', 1),  # Default to 1 if not provided
        'Age': user_input['Age'],
        'HomeSize': user_input['HomeSize'],
        'MonthlyElectricityConsumption': user_input['MonthlyElectricityConsumption'],
        'MonthlyWaterConsumption': user_input['MonthlyWaterConsumption'],
        
        # Location mapping
        'Location_Suburban': 1 if user_input['Location'] == 'Suburban' else 0,
        'Location_Urban': 1 if user_input['Location'] == 'Urban' else 0,
        'Location_Rural': 1 if user_input['Location'] == 'Rural' else 0,

        # DietType mapping
        'DietType_Mostly Animal-Based': 1 if user_input['DietType'] == 'Mostly Animal-Based' else 0,
        'DietType_Mostly Plant-Based': 1 if user_input['DietType'] == 'Mostly Plant-Based' else 0,
        'DietType_Mostly Balanced': 1 if user_input['DietType'] == 'Balanced' else 0,

        # TransportationMode mapping
        'TransportationMode_Car': 1 if user_input['TransportationMode'] == 'Car' else 0,
        'TransportationMode_Public Transit': 1 if user_input['TransportationMode'] == 'Public Transit' else 0,
        'TransportationMode_Walk': 1 if user_input['TransportationMode'] == 'Walk' else 0,
        'TransportationMode_Bike': 1 if user_input['TransportationMode'] == 'Bike' else 0,

        # EnergySource mapping
        'EnergySource_Non-Renewable': 1 if user_input['EnergySource'] == 'Non-Renewable' else 0,
        'EnergySource_Renewable': 1 if user_input['EnergySource'] == 'Renewable' else 0,
        'EnergySource_Mixed': 1 if user_input['EnergySource'] == 'Mixed' else 0,


        # HomeType mapping
        'HomeType_House': 1 if user_input['HomeType'] == 'House' else 0,
        'HomeType_Other': 1 if user_input['HomeType'] == 'Other' else 0,
        'HomeType_Apartment': 1 if user_input['HomeType'] == 'Apartment' else 0,

        # Gender mapping
        'Gender_Male': 1 if user_input['Gender'] == 'Male' else 0,
        'Gender_Female': 1 if user_input['Gender'] == 'Female' else 0,
        'Gender_Non-Binary': 1 if user_input['Gender'] == 'Non-Binary' else 0,
        'Gender_Prefer not to say': 1 if user_input['Gender'] == 'Prefer not to say' else 0,

        # LocalFoodFrequency mapping
        'LocalFoodFrequency_Often': 1 if user_input['LocalFoodFrequency'] == 'Often' else 0,
        'LocalFoodFrequency_Always': 1 if user_input['LocalFoodFrequency'] == 'Always' else 0,
        'LocalFoodFrequency_Rarely': 1 if user_input['LocalFoodFrequency'] == 'Rarely' else 0,
        'LocalFoodFrequency_Sometimes': 1 if user_input['LocalFoodFrequency'] == 'Sometimes' else 0,

        # ClothingFrequency mapping
        'ClothingFrequency_Often': 1 if user_input['ClothingFrequency'] == 'Often' else 0,
        'ClothingFrequency_Always': 1 if user_input['ClothingFrequency'] == 'Always' else 0,
        'ClothingFrequency_Rarely': 1 if user_input['ClothingFrequency'] == 'Rarely' else 0,
        'ClothingFrequency_Sometimes': 1 if user_input['ClothingFrequency'] == 'Sometimes' else 0,

        # SustainableBrands mapping (ensure only one correct mapping)
        'SustainableBrands_TRUE': 1 if user_input['SustainableBrands'] == 'TRUE' else 0,
        'SustainableBrands_FALSE': 1 if user_input['SustainableBrands'] == 'FALSE' else 0,
        
        # EnvironmentalAwareness mapping
        'EnvironmentalAwareness_1': 1 if user_input['EnvironmentalAwareness'] == 1 else 0,
        'EnvironmentalAwareness_2': 1 if user_input['EnvironmentalAwareness'] == 2 else 0,
        'EnvironmentalAwareness_3': 1 if user_input['EnvironmentalAwareness'] == 3 else 0,
        'EnvironmentalAwareness_4': 1 if user_input['EnvironmentalAwareness'] == 4 else 0,
        'EnvironmentalAwareness_5': 1 if user_input['EnvironmentalAwareness'] == 5 else 0,

        # CommunityInvolvement mapping
        'CommunityInvolvement_Low': 1 if user_input['CommunityInvolvement'] == 'Low' else 0,
        'CommunityInvolvement_High': 1 if user_input['CommunityInvolvement'] == 'High' else 0,
        'CommunityInvolvement_None': 1 if user_input['CommunityInvolvement'] == 'None' else 0,
        'CommunityInvolvement_Moderate': 1 if user_input['CommunityInvolvement'] == 'Moderate' else 0,

        # UsingPlasticProducts mapping
        'UsingPlasticProducts_Often': 1 if user_input['UsingPlasticProducts'] == 'Often' else 0,
        'UsingPlasticProducts_Never': 1 if user_input['UsingPlasticProducts'] == 'Never' else 0,
        'UsingPlasticProducts_Rarely': 1 if user_input['UsingPlasticProducts'] == 'Rarely' else 0,
        'UsingPlasticProducts_Sometimes': 1 if user_input['UsingPlasticProducts'] == 'Sometimes' else 0,

        # DisposalMethods mapping
        'DisposalMethods_Composting': 1 if user_input['DisposalMethods'] == 'Composting' else 0,
        'DisposalMethods_Landfill': 1 if user_input['DisposalMethods'] == 'Landfill' else 0,
        'DisposalMethods_Recycling': 1 if user_input['DisposalMethods'] == 'Recycling' else 0,
        'DisposalMethods_Combination': 1 if user_input['DisposalMethods'] == 'Combination' else 0,

        # PhysicalActivities mapping
        'PhysicalActivities_Low': 1 if user_input['PhysicalActivities'] == 'Low' else 0,
        'PhysicalActivities_High': 1 if user_input['PhysicalActivities'] == 'High' else 0,
        'PhysicalActivities_None': 1 if user_input['PhysicalActivities'] == 'None' else 0,
        'PhysicalActivities_Moderate': 1 if user_input['PhysicalActivities'] == 'Moderate' else 0
    }
    
    return transformed_input

def preprocess_input(user_input, scaler):
    """Preprocess user input to match the format and scale used in training."""
    # Map user input
    transformed_input = map_user_input(user_input)

    # Convert the transformed input into a DataFrame
    input_df = pd.DataFrame([transformed_input])

    # Ensure input_df has the same columns as the model
    # Handle missing columns by adding them with a default value of 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default 0

    # Drop any extra columns not present in the model's feature set
    input_df = input_df[model.feature_names_in_]

    # Check if columns like 'TransportationMode_Bike' are missing but should be there
    missing_columns = set(model.feature_names_in_) - set(input_df.columns)
    if missing_columns:
        print("Missing columns:", missing_columns)
        for col in missing_columns:
            input_df[col] = 0  # Add missing columns with default 0

    # Display the DataFrame with all columns
    pd.set_option('display.max_columns', None)  # Show all columns
    print(input_df)

    # Numerical features that need to be scaled
    numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']

    # Check if input_df has all required columns before scaling
    missing_cols = set(numerical_features) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    # Scale numerical features using the scaler loaded
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df


def predict_sustainability(model, user_input, scaler):
    # Preprocess the input data
    preprocessed_input = preprocess_input(user_input, scaler)

    # Make prediction using the preprocessed input
    prediction = model.predict(preprocessed_input)
    
    return prediction[0]


# Example usage with mock input data
user_input = {
    'ParticipantID': 15,
    'Age': 35,
    'HomeSize': 800,
    'MonthlyElectricityConsumption': 100,
    'MonthlyWaterConsumption': 1500,
    'Location': 'Urban',
    'DietType': 'Mostly Plant-Based',
    'TransportationMode': 'Bike',
    'EnergySource': 'Renewable',
    'HomeType': 'Apartment',
    'Gender': 'Female',
    'LocalFoodFrequency': 'Often',
    'ClothingFrequency': 'Rarely',
    'SustainableBrands': 'TRUE',
    'EnvironmentalAwareness': 5,
    'CommunityInvolvement': 'High',
    'UsingPlasticProducts': 'Rarely',
    'DisposalMethods': 'Composting',
    'PhysicalActivities': 'High'
}

prediction = predict_sustainability(model, user_input, scaler)
print(f"Sustainability Rating: {prediction}")
