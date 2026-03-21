import http.server
import os
import socketserver
from pathlib import Path

PORT = 3000


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def main():
    workdir = Path(__file__).parent.resolve()
    os.chdir(workdir)
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving {workdir} on http://0.0.0.0:{PORT}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
