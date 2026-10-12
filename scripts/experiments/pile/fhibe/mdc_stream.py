#!/usr/bin/env python3
"""Stream an MDC dataset archive to stdout, resuming across dropped connections.

R2 closes long transfers part-way (2026-10-09: three attempts died at 35, 33 and
51 GB), so the archive is fetched as sequential 2 GiB Range requests. A broken
request resumes at the exact byte it stopped at, with a fresh signed URL, so the
tar reading stdout never sees the break. Writes the stream's sha256 and the
archive filename to the paths given.

usage: mdc_stream.py DATASET_ID SHA256_OUT NAME_OUT | tar -xz -C DEST
"""

import hashlib
import os
import sys
import time
import requests

API = "https://mozilladatacollective.com/api"
CHUNK = int(os.environ.get("MDC_CHUNK", 2 << 30))  # override only to test
MAX_FAILS = 40  # consecutive failures with no progress before giving up


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


def signed_url(dataset, key):
    r = requests.post(f"{API}/datasets/{dataset}/download", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    d = r.json()
    if "downloadUrl" not in d:
        sys.exit(f"API said: {d}")
    return d["filename"], d["downloadUrl"]


def main():
    dataset, sha_out, name_out = sys.argv[1:4]
    key = open(os.path.expanduser("~/.config/mdc/api_key")).read().strip()
    name, url = signed_url(dataset, key)
    open(name_out, "w").write(name + "\n")
    got_at = time.time()
    h, out = hashlib.sha256(), sys.stdout.buffer
    pos, total, fails, last_log = 0, None, 0, 0.0
    while total is None or pos < total:
        if fails or time.time() - got_at > 6 * 3600:
            time.sleep(min(60, 5 * fails))
            name, url = signed_url(dataset, key)
            got_at = time.time()
        start = pos
        end = pos + CHUNK - 1 if total is None else min(pos + CHUNK, total) - 1
        try:
            with requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=(30, 120)) as r:
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code}")
                total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                for buf in r.iter_content(1 << 20):
                    out.write(buf)
                    h.update(buf)
                    pos += len(buf)
                    if time.time() - last_log > 120:
                        log(f"progress: {pos / 1e9:.1f} / {total / 1e9:.1f} GB")
                        last_log = time.time()
            if pos != end + 1:
                raise IOError(f"short chunk: at {pos}, wanted {end + 1}")
            fails = 0
        except BrokenPipeError:
            sys.exit(f"FAIL: the reader (tar) closed the pipe at byte {pos}")
        except (requests.RequestException, IOError) as e:
            fails = 1 if pos > start else fails + 1
            log(f"resume at {pos} after: {e} (fail {fails})")
            if fails >= MAX_FAILS:
                sys.exit(f"FAIL: {fails} failures in a row at byte {pos}")
    out.flush()
    open(sha_out, "w").write(h.hexdigest() + "\n")
    log(f"stream complete: {pos} bytes")


if __name__ == "__main__":
    main()
