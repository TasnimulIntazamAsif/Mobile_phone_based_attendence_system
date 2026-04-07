from flask import Flask, render_template, request, jsonify
import os, base64, cv2
import numpy as np

from config import DATASET_PATH
from utils.face_utils import encode_faces, recognize_face
from utils.attendance_utils import mark_attendance

app = Flask(__name__)

_LOCALHOSTS = {"127.0.0.1", "::1"}


def _is_local_request() -> bool:
    return request.remote_addr in _LOCALHOSTS


@app.route("/")
def dashboard():
    return render_template("dashboard.html", can_register=_is_local_request())


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    # Security: allow registration only from the same machine (PC),
    # so mobile users can take attendance but cannot register new faces.
    if not _is_local_request():
        return jsonify({"status": "Registration disabled on this device/network"}), 403

    if request.method == "POST":
        data = request.json

        user_id = data.get("user_id")
        name = data.get("name")
        images = data.get("images", [])

        if not user_id or not name or not images:
            return jsonify({"status": "Invalid Data"}), 400

        folder_name = f"{user_id}_{name}"
        user_path = os.path.join(DATASET_PATH, folder_name)

        os.makedirs(user_path, exist_ok=True)

        saved_count = 0

        for i, img_data in enumerate(images):
            try:
                # decode base64
                img_bytes = base64.b64decode(img_data.split(",")[1])
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 🔥 HARD FIX START
                if img is None:
                    print("Decode failed")
                    continue

                # force uint8
                img = np.array(img, dtype=np.uint8)

                # ensure 3 channel
                if len(img.shape) != 3:
                    print("Invalid shape")
                    continue

                # normalize if needed
                if img.max() <= 1:
                    img = (img * 255).astype("uint8")

                # skip too dark
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if gray.mean() < 20:
                    print("Too dark image skipped")
                    continue
                # 🔥 HARD FIX END

                save_path = os.path.join(user_path, f"{i}.jpg")
                cv2.imwrite(save_path, img)

                saved_count += 1

            except Exception as e:
                print("Image Save Error:", e)
                continue

        if saved_count == 0:
            return jsonify({"status": "No valid images captured"}), 400

        encode_faces()

        return jsonify({"status": "Registered Successfully"})

    return render_template("register.html")


# ---------------- ATTENDANCE ----------------
@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        try:
            img_data = request.json["image"]

            img_bytes = base64.b64decode(img_data.split(",")[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"status": "Invalid Image"})

            # Force expected type/shape
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            name, _ = recognize_face(img)

            if name == "No Data":
                return jsonify({"status": "No Users Registered"})

            if name not in ["Unknown", "No Face"]:
                user_id, username = name.split("_", 1)
                status = mark_attendance(user_id, username)

                return jsonify({"status": status, "name": username})

            return jsonify({"status": "Not Matched"})

        except Exception as e:
            print("Attendance Error:", e)
            return jsonify({"status": "Error Occurred"})

    return render_template("attendance.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)