import requests

url = 'http://127.0.0.1:5000'

sent1 = "When comparable rates of flow can be maintained , the results are high ."
sent2 = "The results are high when comparable flow rates can be maintained ."


response = requests.get(url + '/predict',
                        params={'s1': sent1, 's2': sent2})

# response = requests.get(url+'/hello')

print(response.json())
