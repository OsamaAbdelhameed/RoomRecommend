#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install fastapi uvicorn')
get_ipython().system('pip install colabcode')


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# from google.colab import drive
# drive.mount('/content/drive')

# data_path = '/content/drive/MyDrive/UTMSIR Colab files/data.csv'
# labels_path = '/content/drive/MyDrive/UTMSIR Colab files/labels.csv'


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

pickle.dump(knn, open("model_gb.pkl", "wb"))


# In[3]:


from fastapi import FastAPI
from colabcode import ColabCode
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import pickle

cc = ColabCode(port=12000, code=False)

app = FastAPI(title="ML Models as API on Google Colab", description="with FastAPI and ColabCode", version="1.0")


# Define the request body model
class InputData(BaseModel):
    room_type: float
    single_bed: float
    budget: float
    have_transportation: float
    inside_utm: float

knn = None

@app.on_event("startup")
def load_model():
    global knn
    knn = pickle.load(open("model_gb.pkl", "rb"))


@app.get("/")
async def read_root():
  return {"message": "Recommendations requests is working....."}

@app.post("/predict")
async def predict(input_data: InputData):
    # data = dict(input_data)['data']
    print(input_data)
    # Prepare input data for prediction
    input_array = np.array([[input_data.room_type, input_data.single_bed, input_data.budget, input_data.have_transportation, input_data.inside_utm]])

    # Make predictions
    predictions = knn.predict(input_array)
    
    prediction_value = int(predictions[0])
    # Return the prediction result
    # Convert the response content to JSON serializable format
    response_content = jsonable_encoder({"prediction": prediction_value})

    # Return the prediction result
    return response_content


# In[ ]:


cc.run_app(app=app)

