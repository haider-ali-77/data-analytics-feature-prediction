import requests
import pandas as pd
import os


conditions = {
    "Call_To_Action":"SIGN_UP",
    "Logo_Color":"Single",
    "Price_Displayed":"Yes",
    "excluded_features": ["Guarantee_or_Warranty_Mentioned","Subject","Sex","Ad_Format"]
}

r = requests.post('http://0.0.0.0:5001/generate_features',json=conditions)
print(r.text)
