import urllib3
import requests
from requests.exceptions import RequestException
import os

url="https://business.columbia.edu/sites/default/files-efs/imce-uploads/Graham%20Doddsville%20Spring%202023%20Issue%20vFINAL%20(2023.04.27).pdf"

def download_file(url, name):
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            content_disposition=r.headers.get('content-disposition')
            if content_disposition:
                filename = content_disposition.split('filename=')[1]
            else:
                filename = url.split('/')[-1]


            with open(get_path(name), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f'Downloaded: {filename}')
    except RequestException as e:
        print(f'Error downloading {url}: {e}')

# download_file(url, 'john.pdf')

def get_path(name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    new_dir = os.path.join(cur_dir, "prints")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    return os.path.join(new_dir, name)

