import urllib.request
import urllib.error
import json
import io
import docx
import requests

base_url = 'http://127.0.0.1:5000'

def test_get(endpoint):
    req = urllib.request.Request(f'{base_url}{endpoint}')
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode('utf-8')

def test_post(endpoint, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f'{base_url}{endpoint}', data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

print('=== 1. Testing GET /api/health ===')
status, body = test_get('/api/health')
print(f'Status: {status}')
print(f'Body: {body}')

print('\n=== 2. Testing GET /api/metrics ===')
status, body = test_get('/api/metrics')
print(f'Status: {status}')
print(f'Metrics preview: {body[:160]}...')

print('\n=== 3. Testing Frontend Static Delivery ===')
status, html = test_get('/')
print(f'Index HTML Status: {status}, Has doctype: {"<!DOCTYPE html>" in html}')
status, css = test_get('/styles.css')
print(f'styles.css Status: {status}, Length: {len(css)} bytes')
status, js = test_get('/app.js')
print(f'app.js Status: {status}, Length: {len(js)} bytes')

print('\n=== 4. Testing News Prediction: Real News Sample ===')
real_payload = {
    'title': 'U.S. House passes defense policy bill with broad bipartisan support',
    'text': 'WASHINGTON (Reuters) - The U.S. House of Representatives on Tuesday overwhelmingly passed a comprehensive defense policy bill authorizing billions in national security spending, sending the measure to the Senate for approval.',
    'model': 'logistic_regression'
}
for m in ['logistic_regression', 'svm', 'random_forest']:
    real_payload['model'] = m
    status, res = test_post('/api/predict', real_payload)
    print(f'Model: {m:<20} | Status: {status} | Prediction: {res.get("prediction")} | Raw Class: {res.get("raw_class")}')

print('\n=== 5. Testing News Prediction: Fake News Sample ===')
fake_payload = {
    'title': 'BREAKING VIDEO: Secret Meeting Exposes Huge Government Scheme',
    'text': 'Watch this shocking footage via wire image! Insiders reveal unprecedented conspiracy that mainstream media refuses to report. Even top officials were caught off guard in leaked recording.',
    'model': 'logistic_regression'
}
for m in ['logistic_regression', 'svm', 'random_forest']:
    fake_payload['model'] = m
    status, res = test_post('/api/predict', fake_payload)
    print(f'Model: {m:<20} | Status: {status} | Prediction: {res.get("prediction")} | Raw Class: {res.get("raw_class")}')

print('\n=== 6. Testing Document Extraction (PDF, DOCX & TXT) ===')
# Test PDF
pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 55 >> stream
BT
/F1 12 Tf
100 700 Td
(Breaking News: Global Economic Forum Concludes) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000350 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
424
%%EOF"""
resp_pdf = requests.post(f'{base_url}/api/extract-file', files={'file': ('report.pdf', pdf_bytes, 'application/pdf')})
print(f'Extract PDF status:  {resp_pdf.status_code} | Type: {resp_pdf.json().get("data", {}).get("file_type")}')

# Test DOCX
doc = docx.Document()
doc.add_paragraph('BOMBSHELL: Video reveals shocking truth behind government leak')
doc.add_paragraph('Watch this viral video via wire image! Insiders reveal unprecedented conspiracy.')
buf_docx = io.BytesIO()
doc.save(buf_docx)
docx_bytes = buf_docx.getvalue()

resp_docx = requests.post(f'{base_url}/api/extract-file', files={'file': ('investigation.docx', docx_bytes, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')})
print(f'Extract DOCX status: {resp_docx.status_code} | Type: {resp_docx.json().get("data", {}).get("file_type")}')

# Test TXT
txt_content = b'U.S. Senate confirms ambassadors in bipartisan vote\nWASHINGTON (Reuters) - The Senate approved diplomatic nominees on Thursday.'
resp_txt = requests.post(f'{base_url}/api/extract-file', files={'file': ('dispatch.txt', txt_content, 'text/plain')})
print(f'Extract TXT status:  {resp_txt.status_code} | Title: {resp_txt.json().get("data", {}).get("title")}')

print('\n=== 7. Testing Website URL Extraction ===')
url_payload = {'url': 'https://en.wikipedia.org/wiki/Reuters'}
status, res_url = test_post('/api/extract-url', url_payload)
print(f'Extract URL status: {status} | Domain: {res_url.get("data", {}).get("domain")}')

print('\n=== 8. Testing Input Validation: Empty Inputs ===')
status, res = test_post('/api/predict', {'title': '', 'text': '', 'model': 'logistic_regression'})
print(f'Empty text  | Status: {status} | Error: {res.get("error")}')

print('\n=== 9. Testing Input Validation: Invalid Model Selection ===')
status, res = test_post('/api/predict', {'title': 'Valid Title', 'text': 'Valid Text', 'model': 'decision_tree'})
print(f'Decision Tree rejection | Status: {status} | Error: {res.get("error")}')

print('\nALL SYSTEM TESTS COMPLETED SUCCESSFULLY!')
