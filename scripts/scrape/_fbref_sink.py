"""Localhost sink for FBref cup/European fixture lists POSTed from the browser.

FBref blocks headless fetch, so competition schedule pages are opened in a real
browser and each page's PL-team matches are POSTed here. Writes one file per
competition-season to data/raw/external/fbref_fixtures/<comp>_<season>.tsv
(rows: date\thome_slug\taway_slug).
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT_DIR = Path("data/raw/external/fbref_fixtures")
PORT = 8478


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "text/plain"); self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        key, _, tsv = body.partition("\n")
        key = key.strip()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{key}.tsv").write_text(tsv)
        rows = tsv.count("\n") + 1 if tsv.strip() else 0
        print(f"saved {key} ({rows} rows)", flush=True)
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "text/plain"); self.end_headers()
        self.wfile.write(f"saved {key} {rows}".encode())

    def log_message(self, *a):
        return


if __name__ == "__main__":
    print(f"fbref sink on http://localhost:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
