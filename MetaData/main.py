import csv
import requests
import xml.etree.ElementTree as ET

def retrieve_metadata():
    # Implement the logic to retrieve metadata using OAI-PMH
    base_url = 'http://repository.unhas.ac.id/cgi/oai2'
    verb = 'ListRecords'
    metadata_prefix = 'oai_dc'
    # Additional parameters like date range or set can be added if needed
    params = {'verb': verb, 'metadataPrefix': metadata_prefix}
    response = requests.get(base_url, params=params)
    return response.content

def extract_metadata(xml_content):
    # Parse the XML response and extract PDF URLs and document IDs
    root = ET.fromstring(xml_content)
    ns = {'oai': 'http://www.openarchives.org/OAI/2.0/'}
    pdf_urls = []
    document_ids = []
    for record in root.findall('.//oai:record', ns):
        pdf_url = record.find('.//oai:identifier', ns).text
        document_id = record.find('.//oai:identifier', ns).text
        pdf_urls.append(pdf_url)
        document_ids.append(document_id)
    return pdf_urls, document_ids

def write_to_csv(pdf_urls, document_ids, csv_file):
    # Write the PDF URLs and document IDs to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PDF URL', 'Document ID'])
        for url, doc_id in zip(pdf_urls, document_ids):
            writer.writerow([url, doc_id])

# Retrieve metadata using OAI-PMH
metadata_xml = retrieve_metadata()

# Extract PDF URLs and document IDs from the metadata
pdf_urls, document_ids = extract_metadata(metadata_xml)

# Write the extracted information to a CSV file
csv_file = 'metadata.csv'
write_to_csv(pdf_urls, document_ids, csv_file)
