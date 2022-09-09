import requests

url = 'http://127.0.0.1:5000'

sent1 = "Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6 other players representing Britain , " \
        "also on a tour of Australia . "
sent2 = 'Bradd Crellin also represented BARLA Great Britain on a tour through Australia on a tour through Australia ' \
        'with 6 other players representing Cumbria . '

# response = requests.post(url+'/predict',
#                          json={'s1': sent1,  's2': sent2})

response = requests.get(url+'/hello')

print(response.json())
