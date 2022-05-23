import re
import requests
import os

re_art = re.compile(r'\b(a|an|the)\b')	
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')	

def normalize_answer(s):	
    """	
    Lower text and remove punctuation, articles and extra whitespace.	
    """	
    s = s.lower()	
    s = re_punc.sub(' ', s)	
    s = re_art.sub(' ', s)	
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '	
    s = ' '.join(s.split())	
    return s	

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_from_google_drive(gd_id, destination):
    """Use the requests package to download a file from Google Drive."""
    URL = 'https://docs.google.com/uc?export=download'

    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()
