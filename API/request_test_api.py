import requests

url = 'http://127.0.0.1:5000/'
r = requests.get(url)

# GET
print(f'Status code: {r.status_code}')
print(f'Text: \n{r.text}')

# Let's Post something
r1 = requests.post(url, json={'id': '50ME3-B1N'})
print(f'Status code: {r1.status_code}')
print(f'Text: \n{r1.text}')