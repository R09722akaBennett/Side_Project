from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import json

def askAI(data):
    serviceAccountKey = 'references/kddr-demo-eb00eaada6d8.json'
    cloudFuncPath = 'https://us-central1-nongchunxiang-ga4.cloudfunctions.net/chatbot-for-nongchunxiang'
    
    try:
        credentials = service_account.IDTokenCredentials.from_service_account_file(serviceAccountKey, target_audience=cloudFuncPath)
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        jwt = credentials.token
        headers = {
            "Authorization": "Bearer " + jwt,
            "Content-Type": "application/json"
        }
        res = requests.post(cloudFuncPath, headers=headers, data=json.dumps(data))
        return res.json()
    except Exception as e:
        return {"status_code": 500, "answer": str(e)}