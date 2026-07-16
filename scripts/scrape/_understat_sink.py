"""Tiny localhost sink that receives Understat season payloads from the browser.

The Understat league pages sit behind Cloudflare, so they cannot be fetched with
curl. Instead we open them in a real browser (which passes the challenge) and the
page's ``datesData`` JSON is POSTed to this server, which writes each season to
``data/raw/external/understat/EPL_<season>.tsv``.

Run in the background; POST ``season\tTSV`` to ``http://localhost:8477/save``.
CORS is wide-open so a cross-origin ``fetch`` from understat.com is accepted.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT_DIR = Path("data/raw/external/understat")
PORT = 8477


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - health check
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        season, _, tsv = body.partition("\n")
        season = season.strip()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"EPL_{season}.tsv"
        out.write_text(tsv)
        n = tsv.count("\n") + 1 if tsv else 0
        print(f"saved {out} ({n} rows)", flush=True)
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"saved {season} {n}".encode())

    def log_message(self, *args) -> None:  # silence default noise
        return


if __name__ == "__main__":
    print(f"understat sink listening on http://localhost:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
