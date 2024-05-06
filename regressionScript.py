import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('AmesHousing.csv')
print(df.info())

label_encoder = LabelEncoder()

"""for col in df.select_dtypes(include=['object']).columns:
    # Create a new column name by appending "_encoded" to the old column name
    new_col_name = f"{col}_encoded"
    
    # Encode the values in the old column and assign them to the new column
    df[new_col_name] = label_encoder.fit_transform(df[col])
    
    # Drop the old column from the DataFrame
    df.drop(col, axis=1, inplace=True)"""

df['Utilities_encoded'] = label_encoder.fit_transform(df['Utilities'])
df.drop('Utilities', axis=1, inplace=True)
df['Heating_encoded'] = label_encoder.fit_transform(df['Heating'])
df.drop('Heating', axis=1, inplace=True)

columns_to_exclude = ['Heating', 'Utilities']
object_columns = [col for col in df.select_dtypes(include=['object']).columns if col not in columns_to_exclude]

undesired_columns = object_columns + ['SalePrice']
# Split the dataset into features (X) and target variable (y)
X = df[['Year Built', 'Gr Liv Area', 'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Lot Area', 'Utilities_encoded', 'Heating_encoded']]
#X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Reshape the input training data for RNN to include a "fake" time dimension
X_train_imputed_reshaped = X_train_imputed.reshape(X_train_imputed.shape[0], 1, X_train_imputed.shape[1])
X_test_imputed_reshaped = X_test_imputed.reshape(X_test_imputed.shape[0], 1, X_test_imputed.shape[1])

# Define and train the RNN model
model_rnn = Sequential([
    SimpleRNN(units=128, activation='relu', input_shape=(1, X_train_imputed_reshaped.shape[2])),
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=1)
])
model_rnn.compile(optimizer='Adam', loss='mse')
history = model_rnn.fit(X_train_imputed_reshaped, y_train, epochs=30, batch_size=54, validation_split=0.2)

# Evaluate the RNN model
y_pred_rnn = model_rnn.predict(X_test_imputed_reshaped)

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Calculate the range of the target variable
target_range = y.max() - y.min()

# Evaluate the performance of each model
models = {
    'Linear Regression': {'model': LinearRegression(), 'color': 'pink'},
    'Polynomial Regression': {'model': LinearRegression(), 'color': 'green'},
    'Random Forest': {'model': RandomForestRegressor(n_estimators=100, random_state=0), 'color': 'yellow'},
    'RNN': {'model': model_rnn, 'color': 'brown'}
}

for name, model_info in models.items():
    fig, ax = plt.subplots()
    model = model_info['model']
    color = model_info['color']
    
    if name == 'Polynomial Regression':
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train_imputed)
        X_test_poly = poly_features.transform(X_test_imputed)
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    elif name == 'RNN':
        y_pred = y_pred_rnn.flatten()
    else:
        model.fit(X_train_imputed, y_train)
        y_pred = model.predict(X_test_imputed).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Normalize MAE and MSE to the target variable range
    mae_percent = (mae / target_range) * 100
    mse_percent = (mse / (target_range ** 2)) * 100
    
    print(f"{name}:")
    print(f"Mean Absolute Error: {mae_percent:.2f}%")
    print(f"Mean Squared Error: {mse_percent:.2f}%")
    print()
    
    ax.scatter(y_test, y_pred, color=color, label=name)
    ax.plot(y_test, y_test, color='black', label='Actual Prices')
    ax.set_title(name)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()

    plt.tight_layout()
    plt.show()