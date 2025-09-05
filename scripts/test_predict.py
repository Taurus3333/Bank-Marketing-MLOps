import requests
payload = {"instances":[
  {"age":58,"job":"management","marital":"married","education":"tertiary","default":0,
   "balance":2143,"housing":1,"loan":0,"contact":"unknown","day":5,"month":"may",
   "pdays":-1,"previous":0,"poutcome":"unknown","campaign":1}
]}
r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
print("status:", r.status_code)
print("headers:", r.headers)
print("text:")
print(r.text)
