import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
ROOT = os.path.dirname(__file__)
class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path in ("", "/"): path = "/index.html"
        cand = os.path.join(ROOT, "public", path.lstrip("/"))
        if os.path.exists(cand): return cand
        cand2 = os.path.join(ROOT, "static", path.lstrip("/"))
        if os.path.exists(cand2): return cand2
        return super().translate_path(path)
if __name__ == "__main__":
    print("Static site on http://localhost:5500")
    ThreadingHTTPServer(("0.0.0.0", 5500), Handler).serve_forever()
