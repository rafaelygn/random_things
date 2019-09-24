import random

random.getrandbits(32)

hats = {f"hat:{random.getrandbits(32)}": i for i in (
    {
        "color": "black",
        "price": 49.99,
        "style": "fitted",
        "quantity": 1000,
        "npurchased": 0,
    },
    {
        "color": "maroon",
        "price": 59.99,
        "style": "hipster",
        "quantity": 500,
        "npurchased": 0,
    },
    {
        "color": "green",
        "price": 99.99,
        "style": "baseball",
        "quantity": 200,
        "npurchased": 0,
    })
}

for h_id, hat in hats.items():
    print(h_id, hat)
    



import requests
from redis import Redis
from rq import Queue


conn = Redis(host="somehost", port=6379, password="secret")
conn.pipeline()

def count_words_at_url(url):
    '''Here some docstring

    We receive a url and it returns number of letters
    '''

    resp = requests.get(url)
    return len(resp.text.split())


q = Queue(name="some_name", connection=conn)
result = q.enqueue(count_words_at_url, 'http://la.lallala')
 