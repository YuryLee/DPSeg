"""Minimal direct Kaggle MCP transport for the user-authorized submission queue.

Credentials stay in ~/.kaggle/access_token. All connections bypass proxy settings.
The endpoint and upload hosts are restricted; tokens are never included in logs.
"""
import json
import hashlib
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class KaggleError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self):
        self.secret=(Path.home()/'.kaggle/access_token').read_text().strip()
        if not self.secret.startswith('KGAT_'):
            raise KaggleError('Missing Kaggle API token')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        self.next_id=0

    def sanitize(self,text):
        return str(text).replace(self.secret,'[REDACTED]')

    def call(self,name,arguments):
        self.next_id+=1
        payload={'jsonrpc':'2.0','id':self.next_id,'method':'tools/call',
                 'params':{'name':name,'arguments':arguments}}
        request=urllib.request.Request('https://www.kaggle.com/mcp',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json',
            'Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-03-26',
            'Authorization':'Bearer '+self.secret})
        try:
            with self.opener.open(request,timeout=60) as response:
                if 'event-stream' in response.headers.get('Content-Type',''):
                    result=None
                    for _ in range(1000):
                        line=response.readline().decode()
                        if not line:break
                        if line.startswith('data:'):
                            candidate=json.loads(line[5:])
                            if candidate.get('id')==self.next_id:
                                result=candidate;break
                    if result is None:raise KaggleError('No matching MCP response')
                else:result=json.load(response)
        except urllib.error.HTTPError as error:
            detail=self.sanitize(error.read(4000).decode(errors='replace'))
            raise KaggleError(f'Kaggle HTTP {error.code}: {detail}') from None
        if 'error' in result:
            raise KaggleError(self.sanitize(json.dumps(result['error'])))
        value=result['result']
        texts=[x['text'] for x in value.get('content',[]) if x.get('type')=='text']
        if value.get('isError'):
            raise KaggleError(self.sanitize('\n'.join(texts)))
        if len(texts)!=1:raise KaggleError('Unexpected MCP response content')
        try:return json.loads(texts[0])
        except ValueError:raise KaggleError(self.sanitize(texts[0])) from None

    def submissions(self):
        return self.call('search_competition_submissions',{'request':{'competitionName':'rgb-d',
            'sortBy':'Date','group':'All','page':1,'hasPage':True,'pageSize':100,'hasPageSize':True}})['submissions']

    def get_submission(self,reference):
        return self.call('get_competition_submission',{'request':{'ref':reference}})

    def start_upload(self,path):
        path=Path(path)
        return self.call('start_competition_submission_upload',{'request':{
            'competitionName':'rgb-d','hasCompetitionName':True,'contentLength':path.stat().st_size,
            'lastModifiedEpochSeconds':int(path.stat().st_mtime),'fileName':path.name}})

    def submit(self,blob_token,description):
        return self.call('submit_to_competition',{'request':{'competitionName':'rgb-d',
            'blobFileTokens':blob_token,'submissionDescription':description,'hasSubmissionDescription':True}})

    @staticmethod
    def checked_upload_url(url):
        parsed=urllib.parse.urlparse(url)
        allowed=parsed.hostname=='storage.googleapis.com' or (parsed.hostname or '').endswith('.storage.googleapis.com')
        allowed=allowed or (parsed.hostname=='www.googleapis.com'
                            and parsed.path.startswith('/upload/storage/v1/b/kaggle-competitions-submissions/'))
        if parsed.scheme!='https' or not allowed:
            raise KaggleError('Unexpected Kaggle upload destination')
        return url

    def upload_file(self,path,upload,expected_sha256):
        """Kaggle creates the GCS resumable session; PUT the complete CSV to it."""
        path=Path(path)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha256:
                raise KaggleError('CSV changed before upload')
        url=self.checked_upload_url(upload['create_url'])
        # GCS documents storage.googleapis.com as the JSON upload endpoint.
        # Use that verified endpoint when Kaggle returns its legacy www alias.
        parsed=urllib.parse.urlparse(url)
        if parsed.hostname=='www.googleapis.com':
            url=urllib.parse.urlunparse(parsed._replace(netloc='storage.googleapis.com'))
        query=urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if query.get('uploadType')!=['resumable'] or not query.get('upload_id'):
            raise KaggleError('Expected an initialized GCS resumable upload session')
        try:
            with path.open('rb') as stream:
                request=urllib.request.Request(url,data=stream,method='PUT',headers={
                    'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)})
                with self.opener.open(request,timeout=120) as response:
                    if response.status not in (200,201):raise KaggleError('Incomplete file upload')
                    response.read(20000)
        except urllib.error.HTTPError as error:
            raise KaggleError(f'GCS upload failed: HTTP {error.code}') from None
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha256:
                raise KaggleError('CSV changed during upload')
        return upload['token']
