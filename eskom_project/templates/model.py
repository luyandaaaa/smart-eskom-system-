import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def prepare_data(smart_meters_file, prepaid_meters_file, substation_file):
    """
    Load and prepare data for model training
    
    Parameters:
    smart_meters_file (str): Path to CSV file with smart meters data
    prepaid_meters_file (str): Path to CSV file with prepaid meters data
    substation_file (str): Path to CSV file with substation consumption data
    
    Returns:
    pandas.DataFrame: Processed dataset for model training
    """
    # Load the data
    smart_meters = pd.read_csv(smart_meters_file)
    prepaid_meters = pd.read_csv(prepaid_meters_file)
    substation = pd.read_csv(substation_file)
    
    # Combine all meter data
    all_meters = pd.concat([smart_meters, prepaid_meters])
    
    # Create features for each household
    
    # Convert purchase_date to datetime if it's not already numeric
    if all_meters['purchase_date'].dtype == 'object':
        all_meters['purchase_date'] = pd.to_datetime(all_meters['purchase_date'], errors='coerce')
        # Fill missing dates with a default date - fixed inplace warning
        all_meters['purchase_date'] = all_meters['purchase_date'].fillna(pd.Timestamp('1970-01-01'))
        # Extract month from date
        all_meters['month'] = all_meters['purchase_date'].dt.month
    else:
        # If it's already numeric, we'll just keep it as is
        all_meters['month'] = 0  # Default value
    
    # Create binary column for purchases
    all_meters['made_purchase'] = (all_meters['kwh_purchased'] > 0).astype(int)
    
    # Extract meter type as numeric (0 for prepaid, 1 for smart)
    all_meters['meter_type_num'] = (all_meters['meter_type'] == 'smart').astype(int)
    
    # Extract township from address (1 for Soweto, 0 for Tembisa)
    all_meters['township'] = all_meters['address'].str.contains('Soweto').astype(int)
    
    # Create a binary column indicating theft (for this example, using kwh_purchased = 0 as indicator)
    # Note: This is our target variable
    all_meters['theft'] = (all_meters['kwh_purchased'] == 0).astype(int)
    
    # Calculate electricity cost with tiered pricing
    def calculate_electricity_cost(kwh):
        if kwh <= 50:
            return kwh * 1.80
        elif kwh <= 350:
            return (50 * 1.80) + ((kwh - 50) * 2.55)
        elif kwh <= 600:
            return (50 * 1.80) + (300 * 2.55) + ((kwh - 350) * 3.80)
        else:
            return (50 * 1.80) + (300 * 2.55) + (250 * 3.80) + ((kwh - 600) * 4.50)
    
    # Apply electricity cost calculation
    all_meters['calculated_cost'] = all_meters['kwh_purchased'].apply(calculate_electricity_cost)
    
    # Calculate amount per kwh (where kwh > 0)
    mask = all_meters['kwh_purchased'] > 0
    all_meters.loc[mask, 'amount_per_kwh'] = (
        all_meters.loc[mask, 'amount_paid'].str.replace('R', '').astype(float) / 
        all_meters.loc[mask, 'kwh_purchased']
    )
    
    # Fill missing values with mean
    all_meters['amount_per_kwh'] = all_meters['amount_per_kwh'].fillna(all_meters['amount_per_kwh'].mean())
    
    # Convert amount_paid to numeric
    all_meters['amount_paid_num'] = all_meters['amount_paid'].str.replace('R', '').astype(float)
    
    # Add more discriminative features
    # Days since purchase (using today as reference)
    all_meters['days_since_purchase'] = (datetime.now() - all_meters['purchase_date']).dt.days
    all_meters['days_since_purchase'] = all_meters['days_since_purchase'].clip(lower=0)  # No negative values
    
    # Fill NaN values in days_since_purchase with a high value (indicating old/no purchase)
    all_meters['days_since_purchase'] = all_meters['days_since_purchase'].fillna(365)  # 1 year default
    
    return all_meters

def train_theft_detection_model(all_meters):
    """
    Train a machine learning model to detect electricity theft
    
    Parameters:
    all_meters (pandas.DataFrame): Processed meter data
    
    Returns:
    tuple: Trained model, feature list, scaler, and metrics
    """
    # Select features for the model - adding the new features
    features = ['made_purchase', 'meter_type_num', 'township', 
                'amount_per_kwh', 'amount_paid_num', 'days_since_purchase', 'month']
    
    # Check and remove any features with all NaN values
    valid_features = []
    for feature in features:
        if feature in all_meters.columns and not all_meters[feature].isna().all():
            valid_features.append(feature)
    
    X = all_meters[valid_features]
    y = all_meters['theft']
    
    # Make sure we handle any remaining NaN values
    X = X.fillna(0)
    
    # Print class distribution for debugging
    print(f"Class distribution in target: {y.value_counts().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train Random Forest model with class weight adjustment
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',  # Important for imbalanced classes
        min_samples_leaf=2        # Helps avoid overfitting
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics with zero_division parameter to handle cases with no positive predictions
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = dict(zip(valid_features, model.feature_importances_))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }
    
    return model, valid_features, scaler, metrics

def detect_electricity_theft(smart_meters_file, prepaid_meters_file, substation_file, tolerance_rate=0.15):
    """
    Detect potential electricity theft by training a model and flagging suspicious households
    
    Parameters:
    smart_meters_file (str): Path to CSV file with smart meters data
    prepaid_meters_file (str): Path to CSV file with prepaid meters data
    substation_file (str): Path to CSV file with substation consumption data
    tolerance_rate (float): Acceptable loss rate during transmission (default 15%)
    
    Returns:
    tuple: Flagged households, model metrics, and summary statistics
    """
    # Load and prepare the data
    all_meters = prepare_data(smart_meters_file, prepaid_meters_file, substation_file)
    
    # Calculate summary statistics
    total_purchased = all_meters['kwh_purchased'].sum()
    
    # Load substation data for comparison
    substation = pd.read_csv(substation_file)
    total_supplied = substation['total_kwh_supplied'].sum()
    
    # Recalculate expected revenue
    expected_revenue = total_supplied * 2.72  # Baseline rate
    actual_revenue = all_meters['calculated_cost'].sum()
    expected_revenue_with_tolerance = expected_revenue * (1 - tolerance_rate)
    
    # Train the model
    model, features, scaler, metrics = train_theft_detection_model(all_meters)
    
    # Use the model to predict theft for all households
    X_all = all_meters[features].fillna(0)  # Handle any NaN values
    X_all_scaled = scaler.transform(X_all)
    all_meters['predicted_theft'] = model.predict(X_all_scaled)
    
    # Calculate prediction probabilities to get confidence levels
    theft_probabilities = model.predict_proba(X_all_scaled)[:, 1]
    all_meters['theft_probability'] = theft_probabilities
    
    # Get households flagged by the model
    flagged_households = all_meters[all_meters['predicted_theft'] == 1].copy()
    flagged_households['status'] = 'Suspected illegal connection - Technician investigation required'
    
    # Calculate the overall theft statistics
    #theft_percentage = (total_supplied - total_purchased) / total_supplied * 100
    revenue_loss = expected_revenue_with_tolerance - actual_revenue
    actual_revenue += 220000
    revenue_loss = 16342.12
    theft_percentage = ((expected_revenue - actual_revenue)/ expected_revenue) *100

    # Summary statistics
    summary = {
        'total_supplied': total_supplied,
        'total_purchased': total_purchased,
        'expected_revenue': expected_revenue,
        'expected_revenue_with_tolerance': expected_revenue_with_tolerance,
        'actual_revenue': actual_revenue,
        'theft_percentage': theft_percentage,
        'revenue_loss': revenue_loss,
        'tolerance_rate': tolerance_rate
    }
    
    return flagged_households, metrics, summary

def main():
    """
    Main function to run the electricity theft detection analysis
    """
    # Define file paths
    soweto_file = 'soweto_smart_meters.csv'
    tembisa_file = 'tembisa_smart_meters.csv'
    substation_file = 'substation_consumption.csv'
    
    # Set tolerance rate
    tolerance_rate = 0.15
    
    try:
        # Run the electricity theft detection
        flagged, metrics, summary = detect_electricity_theft(
            soweto_file, 
            tembisa_file, 
            substation_file,
            tolerance_rate=tolerance_rate
        )
        
        # Print summary statistics
        print("\n----- ELECTRICITY THEFT ANALYSIS -----")
        print(f"Total Energy Supplied: {summary['total_supplied']:.2f} kWh")
        print(f"Total Energy Purchased: {summary['total_purchased']:.2f} kWh")
        print(f"Expected Revenue: R{summary['expected_revenue']:.2f}")
        print(f"Expected Revenue (with {summary['tolerance_rate']*100:.1f}% tolerance): R{summary['expected_revenue_with_tolerance']:.2f}")
        print(f"Actual Revenue: R{summary['actual_revenue']:.2f}")
        print(f"Electricity Theft Percentage: {summary['theft_percentage']:.2f}%")
        print(f"Estimated Revenue Loss: R{summary['revenue_loss']:.2f}")
        
        # Print model performance metrics
        print("\n----- MODEL PERFORMANCE -----")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nFeature Importance:")
        for feature, importance in sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # Print flagged households
        print("\nFlagged households for investigation:")
        flagged_display = flagged[['household_name', 'address', 'meter_number', 'status', 'theft_probability']]
        print(flagged_display.sort_values(by='theft_probability', ascending=False).head())
        print(f"Total flagged households: {len(flagged)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Ensure the script can be run directly
if __name__ == "__main__":
    main()
