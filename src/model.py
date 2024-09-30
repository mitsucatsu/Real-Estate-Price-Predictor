import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATASET_URL = "https://gitlab.com/cogxen/databank/-/raw/main/2024-09/residential-property-value/real-estate-property-value.csv?ref_type=heads"

def predict_price(distance_to_mrt, num_conv_stores, latitude, longitude):
    # Load the dataset
    re_data = pd.read_csv(DATASET_URL)
    # Selecting features and target variables
    features = [
        "Distance to the nearest MRT station",
        "Number of convinience stores",
        "Latitude",
        "Longitude"
    ]
    target = [
        "House price of unit area"
    ]

    x = re_data[features]
    y = re_data[target]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Training the model
    model= LinearRegression()
    model.fit(x, y)

    # Predicting with the model
    y_pred = model.predict(X_test)

    # Calculate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Baseline
    y_mean_pred = [y_train.mean()] * len(y_test)
    baseline_mse = mean_squared_error(y_test, y_mean_pred)

    # Create a DataFrame for the input features
    input_data = pd.DataFrame(
        [[distance_to_mrt, num_conv_stores, latitude, longitude]],
        columns=features
    )
    
    # Predict the price
    price= model.predict(input_data)[0][0]

    print(price)

    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    print(f"BASELINE MSE: {baseline_mse}")

predict_price(423, 69, 525.60, 420)
