from app.main import bp
from flask.json import jsonify
from flask import Response, render_template, request
from app.data.utils import (
    cal_vibration, 
    track_box, 
    detecting, 
    messuring_l, 
    messuring_r, 
    dial_plate_l,
    dial_plate_r,
    dial_plate_square,
    qrcode,
    characters_detection,
    face_rec,
    close_video,
)
from app import redis_client
import json


@bp.route("/turn_off")
def turn_off():
    close_video()
    return jsonify("video turned off")


@bp.route("/vib")
def inedx():
    return render_template("vibration_messure.html")


@bp.route("/video_feed_vib")
def video_feed_vib():
    return Response(track_box(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/write-bbox/<key>", methods=["POST"])
def write_bbox(key):
    received_data = request.json
    redis_client.hset(f"rect", key, json.dumps(received_data))
    return jsonify("OK")


@bp.route("/vib-chart")
def vib_chart():
    return cal_vibration()


@bp.route("/refresh", methods=["POST"])
def refresh():
    redis_client.hset("refresh", 1, 1)
    redis_client.expire("refresh", 1)
    return jsonify("OK")


@bp.route("/write-len", methods=["POST"])
def write_len():
    received_data = request.json
    redis_client.hset(f"rect", "real", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/human")
def human():
    return render_template("human_detection.html")


@bp.route("/video_feed_hd")
def video_feed_hd():
    return Response(detecting(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/length")
def length():
    return render_template("length_messure.html")


@bp.route("/video_feed_lw")
def video_feed_lw_l():
    return Response(messuring_l(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/video_feed_raw")
def video_feed_lw_r():
    return Response(messuring_r(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/write-filter", methods=["POST"])
def write_filter():
    received_data = request.json
    redis_client.hset(f"rect", "filter", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/write-threshold", methods=["POST"])
def write_thresold():
    received_data = request.json
    redis_client.hset(f"rect", "threshold", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/write-scale", methods=["POST"])
def write_scale():
    received_data = request.json
    redis_client.hset(f"rect", "scale", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/dial")
def dial():
    return render_template("read_dial.html")


@bp.route("/video_feed_dp_l")
def video_feed_dp_l():
    return Response(dial_plate_l(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/video_feed_dp_r")
def video_feed_dp_r():
    return Response(dial_plate_r(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/write-circles-dial", methods=["POST"])
def write_dimension():
    received_data = request.json
    redis_client.hset(f"rect", "minRadius_dial", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/write-lines-dial", methods=["POST"])
def write_thresold_dial():
    received_data = request.json
    redis_client.hset(f"rect", "lines_dial", json.dumps(received_data))
    return jsonify("OK")


@bp.route("/dial-sq")
def dial_sq():
    return render_template("read_dial_square.html")


@bp.route("/video_feed_dp_square")
def video_feed_dp_square():
    return Response(dial_plate_square(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/qr")
def qr():
    return render_template("qrcode.html")


@bp.route("/video_feed_qr")
def video_feed_qr():
    return Response(qrcode(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/ocr")
def ocr():
    return render_template("ocr.html")


@bp.route("/video_feed_ocr")
def video_feed_ocr():
    return Response(characters_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/face")
def face():
    return render_template("face_recognition.html")


@bp.route("/video_feed_face")
def video_feed_face():
    return Response(face_rec(), mimetype='multipart/x-mixed-replace; boundary=frame')