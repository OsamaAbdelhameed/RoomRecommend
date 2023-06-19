import os
import uvicorn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

# Load your dataset
X = pd.read_csv('./data.csv', header=None)
y = pd.read_csv('./labels.csv', header=None)
y = y.iloc[:, 0]

# Convert non-numeric values to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
X.dropna(inplace=True)
y = y[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2) 

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

app = FastAPI(title="ML Models as API on Google Colab", description="with FastAPI and ColabCode", version="1.0")

# Define the request body model
class InputData(BaseModel):
    room_type: float
    single_bed: float
    budget: float
    have_transportation: float
    inside_utm: float

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Recommendations requests is working....."}

@app.post("/predict")
async def predict(input_data: InputData):
    print(input_data)
    # Prepare input data for prediction
    input_array = pd.Series([input_data.room_type, input_data.single_bed, input_data.budget, input_data.have_transportation, input_data.inside_utm]).values.reshape(1, -1)

    # Make predictions
    predictions = knn.predict(input_array)

    prediction_value = int(predictions[0])
    # Return the prediction result
    # Convert the response content to JSON serializable format
    response_content = jsonable_encoder({"prediction": prediction_value})

    # Return the prediction result
    return response_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
