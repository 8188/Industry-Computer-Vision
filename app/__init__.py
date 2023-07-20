from flask import Flask
from config import Config
from flask_cors import CORS
from flask_redis import FlaskRedis
import logging
from logging.handlers import RotatingFileHandler
import os
import warnings
import sys
import atexit
import cv2
warnings.filterwarnings("ignore")


def sigint_handler(sig, frame):
    # 清理
    logger.info('Exit by KeyboardInterrupt')
    sys.exit(0)


def sigterm_handler(sig, frame):
    logger.info('Terminated by Kill')


redis_client = FlaskRedis()
cors = CORS()
logger = logging.getLogger(__name__)
video = None


def cleanup():
    print("---------clean up---------")
    redis_client.connection_pool.disconnect()
    if video is not None:
        video.release()
        cv2.destroyAllWindows()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    cors.init_app(app, supports_credentials=True)

    redis_client.init_app(app)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/flask.log', maxBytes=10240,
            backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        logger.info('Startup')   

    atexit.register(cleanup)

    return app