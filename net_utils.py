#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import get_config

def make_session():
    s = requests.Session()
    s.headers.update({"Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"})
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()
HTTP_TIMEOUT = float(get_config().get("HTTP_TIMEOUT", "30"))
