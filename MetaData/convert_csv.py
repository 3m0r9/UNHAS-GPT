import requests
import pandas as pd

url = "http://repository.unhas.ac.id/cgi/search/archive/advanced/export_unhaseprints_JSON.js?screen=Search&dataset=archive&_action_export=1&output=JSON&exp=0%7C1%7C-date%2Fcreators_name%2Ftitle%7Carchive%7C-%7Cdocuments.format%3Adocuments.format%3AANY%3AEQ%3Atext%7Cispublished%3Aispublished%3AANY%3AEQ%3Apub%7Ctype%3Atype%3AANY%3AEQ%3Aarticle+thesis+experiment+teaching_resource%7C-%7Ceprint_status%3Aeprint_status%3AANY%3AEQ%3Aarchive%7Cmetadata_visibility%3Ametadata_visibility%3AANY%3AEQ%3Ashow&n=&cache=128568"

# Fetch the JSON data from the URL
response = requests.get(url)
json_data = response.json()

# Convert the JSON data to a DataFrame
data = pd.DataFrame(json_data)

# Convert the DataFrame to a CSV file
data.to_csv('metadata_JSON_URL.csv', index=False)
