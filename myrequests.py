import requests
import json

url = 'https://flask-test-10.herokuapp.com/invocations'

data = {
        'Pclass':[3,3,3],
        'Sex': ['male', 'female', 'male'],
        'Age':[4, 22, 28]
    }
j_data = json.dumps(data)

headers = {'Content-Type': 'application/json'}
print("Sending request for model...")
print(f"Data: {j_data}")
r = requests.post(url, json=j_data, headers=headers)
print(f"Response: {r.text}")
