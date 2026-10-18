"""Bounded Xet byte-range reader; HTTPS only, no environment proxies or disk tokens.

Protocol references:
https://huggingface.co/docs/xet/api
https://huggingface.co/docs/xet/en/download-protocol
https://huggingface.co/docs/xet/xorb

Partial reads validate HTTP ranges, chunk framing, decompressed and term lengths.
They cannot verify the complete file's Merkle hash without reading the whole file.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import ctypes.util
import email.policy
from email.parser import BytesParser
import json
import re
import time
from urllib.parse import quote, urlsplit

import requests
import urllib3


class RangeReadError(RuntimeError):
    """Sanitized failure: never includes signed URLs or bearer tokens."""


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise RangeReadError('Invalid integer in reconstruction metadata')
    return value


def _lz4_frame(payload, expected):
    library = ctypes.util.find_library('lz4')
    if not library:
        raise RangeReadError('System liblz4 is required for compressed chunks')
    lib = ctypes.CDLL(library)
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    lib.LZ4F_createDecompressionContext.argtypes = [ctypes.POINTER(ptr), ctypes.c_uint]
    lib.LZ4F_createDecompressionContext.restype = size
    lib.LZ4F_decompress.argtypes = [ptr, ptr, ctypes.POINTER(size), ptr, ctypes.POINTER(size), ptr]
    lib.LZ4F_decompress.restype = size
    lib.LZ4F_freeDecompressionContext.argtypes = [ptr]
    lib.LZ4F_freeDecompressionContext.restype = size
    lib.LZ4F_isError.argtypes = [size]
    lib.LZ4F_isError.restype = ctypes.c_uint
    context = ptr()
    if lib.LZ4F_isError(lib.LZ4F_createDecompressionContext(ctypes.byref(context), 100)):
        raise RangeReadError('LZ4 decoder initialization failed')
    source = ctypes.create_string_buffer(payload)
    target = ctypes.create_string_buffer(expected + 1)
    consumed = produced = 0
    try:
        while True:
            src_size, dst_size = size(len(payload) - consumed), size(expected + 1 - produced)
            code = lib.LZ4F_decompress(context, ctypes.byref(target, produced), ctypes.byref(dst_size),
                                      ctypes.byref(source, consumed), ctypes.byref(src_size), None)
            if lib.LZ4F_isError(code):
                raise RangeReadError('Invalid LZ4 frame')
            consumed += src_size.value
            produced += dst_size.value
            if produced > expected:
                raise RangeReadError('LZ4 decoded size exceeds chunk header')
            if code == 0:
                break
            if not src_size.value and not dst_size.value:
                raise RangeReadError('Truncated or oversized LZ4 frame')
        if consumed != len(payload) or produced != expected:
            raise RangeReadError('LZ4 decoded length mismatch or trailing data')
        return target.raw[:produced]
    finally:
        lib.LZ4F_freeDecompressionContext(context)


def decode_chunks(body, count):
    """Decode exactly count chunks, rejecting trailing and malformed bytes."""
    chunks = []
    cursor = 0
    for _ in range(_integer(count, 1)):
        if cursor + 8 > len(body):
            raise RangeReadError('Truncated xorb chunk header')
        header = body[cursor:cursor + 8]
        compressed = int.from_bytes(header[1:4], 'little')
        scheme = header[4]
        expanded = int.from_bytes(header[5:8], 'little')
        if header[0] != 0 or scheme not in (0, 1, 2) or not 0 < expanded <= 131072 or not 0 < compressed <= 131072:
            raise RangeReadError('Unsupported or invalid xorb chunk header')
        cursor += 8
        if cursor + compressed > len(body):
            raise RangeReadError('Truncated xorb chunk payload')
        payload = body[cursor:cursor + compressed]
        cursor += compressed
        decoded = payload if scheme == 0 else _lz4_frame(payload, expanded)
        if len(decoded) != expanded:
            raise RangeReadError('Xorb chunk length mismatch')
        if scheme == 2:
            ungrouped, offset = bytearray(expanded), 0
            for group in range(4):
                length = (expanded + 3 - group) // 4
                ungrouped[group::4] = decoded[offset:offset + length]
                offset += length
            decoded = bytes(ungrouped)
        chunks.append(decoded)
    if cursor != len(body):
        raise RangeReadError('Unexpected trailing bytes in xorb range')
    return chunks


class RangeReader:
    """read_range(start, length) returns bytes; never downloads an entire file.

    max_download_bytes bounds cumulative response-body bytes for this instance.
    A caller requesting an authorized larger subset may explicitly increase it.
    max_read_bytes bounds an individual requested uncompressed range.
    """
    def __init__(self, repo, revision, file_hash, size, *, max_download_bytes=32 * 1024 * 1024,
                 max_read_bytes=16 * 1024 * 1024, max_cache_bytes=32 * 1024 * 1024,
                 max_network_bytes=None, endpoint='https://hf-mirror.com'):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('A repository owner/name and pinned 40-character revision are required')
        if not re.fullmatch(r'[0-9a-f]{64}', file_hash):
            raise ValueError('Expected a 64-character Xet file hash')
        if endpoint != 'https://hf-mirror.com':
            raise ValueError('Only the configured public mirror endpoint is permitted')
        self.repo, self.revision, self.file_hash = repo, revision, file_hash
        self.size = _integer(size, 1)
        self.max_download_bytes = _integer(max_download_bytes if max_network_bytes is None else max_network_bytes, 1)
        self.max_cache_bytes = _integer(max_cache_bytes)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.max_read_bytes = _integer(max_read_bytes, 1)
        self.downloaded_bytes = 0
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = '/etc/ssl/certs/ca-certificates.crt'
        self.session.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept-Encoding': 'identity'})
        self._token = self._cas_url = None
        self._expires = 0

    def close(self):
        self._token = None
        self._cache.clear()
        self._cache_bytes = 0
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _request(self, url, *, headers=None, limit=2 * 1024 * 1024, allow_missing=False):
        parsed = urlsplit(url)
        host = parsed.hostname or ''
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise RangeReadError('Refusing a non-HTTPS or credential-bearing endpoint')
        if host != 'hf-mirror.com' and not host.endswith('.xethub.hf.co') and not host.endswith('.cdn.hf.co'):
            raise RangeReadError('Unexpected Xet endpoint hostname')
        remaining = self.max_download_bytes - self.downloaded_bytes
        if remaining <= 0:
            raise RangeReadError('Cumulative download budget exhausted')
        limit = min(limit, remaining)
        try:
            with self.session.get(url, headers=headers, timeout=(15, 45), stream=True, allow_redirects=False) as response:
                if allow_missing and response.status_code in (404, 501):
                    return response.status_code, {}, b''
                if response.status_code not in (200, 206):
                    raise RangeReadError(f'HTTP status {response.status_code} from {host}')
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > limit:
                    raise RangeReadError('Response exceeds bounded download allowance')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise RangeReadError('Unexpected HTTP content encoding')
                body = bytearray()
                # Never read beyond the allowance, including unknown-length responses.
                while len(body) < limit:
                    chunk = response.raw.read(min(65536, limit - len(body)))
                    if not chunk:
                        break
                    self.downloaded_bytes += len(chunk)
                    body.extend(chunk)
                if len(body) == limit and (not content_length or int(content_length) != len(body)):
                    raise RangeReadError('Unknown-length response reached the download allowance')
                return response.status_code, dict(response.headers), bytes(body)
        except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
            raise RangeReadError(f'HTTPS transport failed for {host}: {type(exc).__name__}') from None

    def _authorize(self):
        if self._token and self._expires > time.time() + 30:
            return
        url = f'{self.endpoint}/api/datasets/{quote(self.repo, safe="/")}/xet-read-token/{self.revision}'
        # A public mirror may cache a token past expiry; request fresh metadata.
        url += f'?refresh={time.time_ns()}'
        _, _, body = self._request(url, headers={'Cache-Control': 'no-cache'})
        try:
            metadata = json.loads(body)
            token, cas_url = metadata['accessToken'], metadata['casUrl']
            expires = float(metadata['exp'])
        except (ValueError, KeyError, TypeError):
            raise RangeReadError('Malformed Xet authorization metadata') from None
        parsed = urlsplit(cas_url)
        if parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.xethub.hf.co') or parsed.query or parsed.fragment:
            raise RangeReadError('Unexpected CAS service URL')
        if not isinstance(token, str) or not token or expires <= time.time():
            raise RangeReadError('Invalid or expired Xet authorization')
        self._token, self._cas_url, self._expires = token, cas_url.rstrip('/'), expires

    @staticmethod
    def _validate_range(headers, start, end):
        value = next((v for k, v in headers.items() if k.lower() == 'content-range'), '')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', value)
        if not match or int(match[1]) != start or int(match[2]) != end:
            raise RangeReadError('Missing or incorrect HTTP Content-Range')
        if match[3] != '*' and int(match[3]) <= end:
            raise RangeReadError('Inconsistent HTTP resource length')

    def _fetch(self, entry):
        descriptors = entry['ranges']
        if not isinstance(descriptors, list) or not descriptors:
            raise RangeReadError('Empty xorb range metadata')
        total = 0
        parts = []
        for desc in descriptors:
            start, end = _integer(desc['bytes']['start']), _integer(desc['bytes']['end'])
            cs, ce = _integer(desc['chunks']['start']), _integer(desc['chunks']['end'])
            if end < start or ce <= cs or ce - cs > 8192:
                raise RangeReadError('Invalid xorb bounds')
            total += end - start + 1
            parts.append(f'{start}-{end}')
        status, headers, body = self._request(entry['url'], headers={'Range': 'bytes=' + ','.join(parts)},
                                              limit=total + 65536)
        if status != 206:
            raise RangeReadError('Xorb server did not honor the byte range')
        if len(descriptors) == 1:
            self._validate_range(headers, descriptors[0]['bytes']['start'], descriptors[0]['bytes']['end'])
            bodies = [body]
        else:
            content_type = next((v for k, v in headers.items() if k.lower() == 'content-type'), '')
            if not content_type.lower().startswith('multipart/byteranges;'):
                raise RangeReadError('Expected multipart byte-range response')
            parsed = BytesParser(policy=email.policy.default).parsebytes(
                ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
            messages = list(parsed.iter_parts())
            if len(messages) != len(descriptors):
                raise RangeReadError('Multipart range count mismatch')
            bodies = []
            for message, desc in zip(messages, descriptors):
                self._validate_range(dict(message.items()), desc['bytes']['start'], desc['bytes']['end'])
                bodies.append(message.get_payload(decode=True))
        chunks = {}
        for desc, data in zip(descriptors, bodies):
            if len(data) != desc['bytes']['end'] - desc['bytes']['start'] + 1:
                raise RangeReadError('Xorb range payload length mismatch')
            cs, ce = desc['chunks']['start'], desc['chunks']['end']
            chunks.update(zip(range(cs, ce), decode_chunks(data, ce - cs)))
        return chunks

    def read_range(self, start, length):
        start, length = _integer(start), _integer(length)
        if length > self.max_read_bytes or start + length > self.size:
            raise ValueError('Requested range exceeds file bounds or per-read limit')
        if length == 0:
            return b''
        for cached_start, cached in list(self._cache.items()):
            relative = start - cached_start
            if 0 <= relative and relative + length <= len(cached):
                self._cache.move_to_end(cached_start)
                return cached[relative:relative + length]
        self._authorize()
        headers = {'Authorization': 'Bearer ' + self._token, 'Range': f'bytes={start}-{start + length - 1}'}
        base = f'{self._cas_url}/v2/reconstructions/{self.file_hash}'
        status, _, body = self._request(base, headers=headers, allow_missing=True)
        if status in (404, 501):
            _, _, body = self._request(f'{self._cas_url}/v1/reconstructions/{self.file_hash}', headers=headers)
        try:
            reconstruction = json.loads(body)
            terms = reconstruction['terms']
            offset = _integer(reconstruction['offset_into_first_range'])
            if offset > start:
                raise RangeReadError('Reconstruction offset precedes file start')
            if not isinstance(terms, list) or not terms:
                raise RangeReadError('Empty reconstruction terms')
            declared = sum(_integer(term['unpacked_length'], 1) for term in terms)
            # A server ignoring Range must not cause an unbounded reconstruction.
            if declared < offset + length or declared > length + 2 * 131072:
                raise RangeReadError('Reconstruction does not match requested byte range')
            xorbs = reconstruction.get('xorbs')
            if xorbs is None:
                xorbs = {}
                for hash_, entries in reconstruction['fetch_info'].items():
                    xorbs[hash_] = [{'url': entry['url'], 'ranges': [
                        {'chunks': entry['range'], 'bytes': entry['url_range']}]} for entry in entries]
            cache, assembled = {}, []
            for term in terms:
                hash_ = term['hash']
                if not re.fullmatch(r'[0-9a-f]{64}', hash_):
                    raise RangeReadError('Invalid xorb identifier')
                if hash_ not in cache:
                    cache[hash_] = {}
                    for entry in xorbs[hash_]:
                        cache[hash_].update(self._fetch(entry))
                cs, ce = _integer(term['range']['start']), _integer(term['range']['end'])
                if ce <= cs or ce - cs > 8192:
                    raise RangeReadError('Invalid term chunk range')
                decoded = b''.join(cache[hash_][i] for i in range(cs, ce))
                if len(decoded) != term['unpacked_length']:
                    raise RangeReadError('Reconstruction term length mismatch')
                assembled.append(decoded)
            complete = b''.join(assembled)
            data = complete[offset:offset + length]
            if len(data) != length:
                raise RangeReadError('Incomplete requested range')
            cache_start = start - offset
            if cache_start >= 0 and len(complete) <= self.max_cache_bytes:
                previous = self._cache.pop(cache_start, b'')
                self._cache_bytes -= len(previous)
                while self._cache and self._cache_bytes + len(complete) > self.max_cache_bytes:
                    _, discarded = self._cache.popitem(last=False)
                    self._cache_bytes -= len(discarded)
                self._cache[cache_start] = complete
                self._cache_bytes += len(complete)
            return data
        except (KeyError, TypeError, ValueError, IndexError):
            raise RangeReadError('Malformed reconstruction metadata') from None
