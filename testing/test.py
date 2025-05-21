from apis.apis import app
from fastapi.testclient import TestClient



client = TestClient(app)



def test_home_endpoint():
    
    response = client.get('/')

    assert response.status_code == 200
    assert "message" in response.json()




def test_health_endpoint():
    
    response = client.get('/health')

    assert response.status_code == 200
    assert "status" in response.json()




def test_predict_endpoint():

    # create dummy data
    dummy_data = {
                                'CustomerId': [1579233, 1579234],
                                'Surname': ['Smith', 'Johnson'],
                                'CreditScore': [600, 720],
                                'Geography': ['France', 'Germany'],
                                'Gender': ['Male', 'Female'],
                                'Age': [40, 30],
                                'Tenure': [5, 2],
                                'Balance': [100000.0, 200000.0],
                                'NumOfProducts': [1, 2],
                                'HasCrCard': [1, 0],
                                'IsActiveMember': [1, 1],
                                'EstimatedSalary': [50000.0, 60000.0]
                }
    

    response = client.post('/predict', json=dummy_data)

    assert response.status_code == 200
    assert "prediction" in response.json()


