from app import create_app, sigint_handler, sigterm_handler
from gevent import pywsgi, monkey
import signal


monkey.patch_all()

app = create_app()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler) # windows not support
    pywsgi.WSGIServer(("0.0.0.0", 8080), app).serve_forever()