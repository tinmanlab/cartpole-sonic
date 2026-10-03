#!/usr/bin/env python3
import json, os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT=int(os.environ.get("PORT","8765"))
LATEST={"status":"no-browser"}

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control","no-store")
        self.send_header("Cross-Origin-Opener-Policy","same-origin")
        super().end_headers()

    def do_GET(self):
        if self.path=="/telemetry":
            body=json.dumps(LATEST,separators=(",",":")).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path!="/telemetry":
            self.send_error(404); return
        try:
            n=min(int(self.headers.get("Content-Length","0")),65536)
            data=json.loads(self.rfile.read(n).decode() or "{}")
            LATEST.clear(); LATEST.update(data)
            body=b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(400,str(e))

if __name__=="__main__":
    print(f"CartPole Latent Lab: http://localhost:{PORT}",flush=True)
    ThreadingHTTPServer(("0.0.0.0",PORT),Handler).serve_forever()
