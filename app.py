import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import shutil
import zipfile
import cv2
import numpy as np
import logging
import time
from itertools import combinations
from deepface import DeepFace
from werkzeug.utils import secure_filename
import hashlib
import pickle
from flask import Flask, request, render_template, send_file, redirect, url_for, send_from_directory, jsonify
import base64

# ===== Logging Setup =====
logging.basicConfig(
    filename="face_sortir.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ===== Konfigurasi =====
input_folder = "foto_input"
reference_folder = "foto_referensi"
output_folder = "hasil_sortir"
output_blur = "foto_blur"

MODEL = "ArcFace"  # Default
MODEL_LIST = ["ArcFace", "Facenet"]  # Hapus VGG-Face dari pilihan model
DETECTOR = "retinaface"
BLUR_THRESHOLD = 50
SIM_THRESHOLD_DEFAULT = 0.75

# ===== Setup Folder =====
os.makedirs(input_folder, exist_ok=True)
os.makedirs(reference_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_blur, exist_ok=True)

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# ===== Util =====
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def is_image(p): 
    return p.lower().endswith(IMG_EXT)

def l2(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x) + 1e-8
    return x / n

def cos_sim(a, b):
    a = l2(a); b = l2(b)
    return float(np.dot(a, b))

def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def safe_copy(src, dst):
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logging.error(f"Failed to copy {src} to {dst}: {e}")

def folder_hash(folder):
    h = hashlib.md5()
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if is_image(f):
                path = os.path.join(root, f)
                h.update(path.encode())
                h.update(str(os.path.getmtime(path)).encode())
    return h.hexdigest()

# ===== Reference Embeddings =====
def build_reference_embeddings(model=MODEL, detector=DETECTOR):
    try:
        folder_sig = folder_hash(reference_folder)
        cache_file = f"ref_embeds_{model}_{detector}_{folder_sig}.pkl"

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    ref_embeds, ref_centroids = pickle.load(f)
                logging.info("Loaded reference embeddings from cache.")
                return ref_embeds, ref_centroids
            except Exception as e:
                logging.error(f"Failed to load cache: {e}")

        ref_embeds = {}
        ref_centroids = {}
        has_ref_file = False
        download_failed = False
        download_error = ""

        for person in os.listdir(reference_folder):
            person_dir = os.path.join(reference_folder, person)
            if not os.path.isdir(person_dir): 
                continue
            embs = []
            for f in os.listdir(person_dir):
                p = os.path.join(person_dir, f)
                if not is_image(p): 
                    continue
                has_ref_file = True
                try:
                    reps = DeepFace.represent(img_path=p, model_name=model, detector_backend=detector, enforce_detection=False)
                    if reps:
                        embs.append(l2(reps[0]["embedding"]))
                except Exception as e:
                    logging.error(f"Gagal {person}/{f}: {e}")
                    if "No such file" in str(e) or "download" in str(e).lower() or "URL fetch failure" in str(e):
                        download_failed = True
                        download_error = str(e)
            if embs:
                ref_embeds[person] = embs
                ref_centroids[person] = l2(np.mean(embs, axis=0))

        if not has_ref_file:
            raise RuntimeError("Tidak ada foto referensi. Silakan upload foto referensi terlebih dahulu.")

        if download_failed:
            raise RuntimeError(f"Model {model} belum berhasil didownload. Pastikan koneksi internet stabil. Detail: {download_error}")

        with open(cache_file, "wb") as f:
            pickle.dump((ref_embeds, ref_centroids), f)
        logging.info("Saved reference embeddings to cache.")
        return ref_embeds, ref_centroids
    except Exception as e:
        raise RuntimeError(str(e))

def auto_threshold(ref_embeds):
    intra_sims = []
    for person, embs in ref_embeds.items():
        for a, b in combinations(embs, 2):
            intra_sims.append(cos_sim(a, b))
    if intra_sims:
        p10 = float(np.percentile(intra_sims, 10))
        return float(np.clip(p10 - 0.03, 0.60, 0.90))
    else:
        return SIM_THRESHOLD_DEFAULT

def process_input_images(ref_centroids, sim_threshold, blur_threshold, model=MODEL, detector=DETECTOR):
    counts = {"total": 0, "blur": 0, "lainnya": 0}
    per_person = {k: 0 for k in ref_centroids.keys()}
    lainnya_dir = os.path.join(output_folder, "lainnya")
    os.makedirs(lainnya_dir, exist_ok=True)

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)
        if not is_image(path): 
            continue
        counts["total"] += 1

        if is_blurry(path, threshold=blur_threshold):
            safe_copy(path, os.path.join(output_blur, file))
            counts["blur"] += 1
            continue

        try:
            reps = DeepFace.represent(img_path=path, model_name=model, detector_backend=detector, enforce_detection=False)
        except Exception as e:
            logging.error(f"DeepFace error on {file}: {e}")
            reps = []

        if not reps:
            shutil.copy2(path, os.path.join(lainnya_dir, file))
            counts["lainnya"] += 1
            continue

        matched_people = set()
        for rep in reps:
            face_emb = l2(rep["embedding"])
            for name, centroid in ref_centroids.items():
                sim = cos_sim(face_emb, centroid)
                if sim >= sim_threshold:
                    matched_people.add(name)

        if matched_people:
            for name in matched_people:
                dest = os.path.join(output_folder, name)
                os.makedirs(dest, exist_ok=True)
                base, ext = os.path.splitext(secure_filename(file))
                unique_name = f"{base}_{int(time.time()*1000)}{ext}"
                target = os.path.join(dest, unique_name)
                shutil.copy2(path, target)
                per_person[name] += 1
        else:
            shutil.copy2(path, os.path.join(lainnya_dir, file))
            counts["lainnya"] += 1

    return counts, per_person

def run_sorting(model=MODEL, detector=DETECTOR, blur_threshold=BLUR_THRESHOLD):
    start_time = time.time()
    reset_folder(output_folder)
    reset_folder(output_blur)

    ref_embeds, ref_centroids = build_reference_embeddings(model, detector)
    if not ref_centroids:
        raise RuntimeError("Tidak ada embedding referensi.")

    sim_threshold = auto_threshold(ref_embeds)
    counts, per_person = process_input_images(ref_centroids, sim_threshold, blur_threshold, model, detector)

    elapsed = time.time() - start_time
    logging.info(f"Sorting completed in {elapsed:.2f}s. Counts: {counts}")
    return counts, per_person, sim_threshold

# ===== Web App =====
app = Flask(__name__)

@app.route("/")
def index():
    people = {}
    for person in os.listdir(reference_folder):
        person_dir = os.path.join(reference_folder, person)
        if not os.path.isdir(person_dir):
            continue
        people[person] = os.listdir(person_dir)

    input_files = [f for f in os.listdir(input_folder) if is_image(f)]

    return render_template(
        "index.html",
        people=people,
        input_files=input_files,
        model_list=MODEL_LIST,
        selected_model=MODEL,
        blur_threshold=BLUR_THRESHOLD
    )

@app.route("/upload_ref", methods=["POST"])
def upload_ref():
    person_name = request.form.get("person_name")
    files = request.files.getlist("files")
    warning_msg = None

    if not person_name:
        warning_msg = "Nama orang wajib diisi."
    elif not files or all(f.filename == "" for f in files):
        warning_msg = f"Tidak ada file yang dipilih untuk {person_name}."
    else:
        save_dir = os.path.join(reference_folder, secure_filename(person_name))
        os.makedirs(save_dir, exist_ok=True)
        for f in files:
            if f.filename == "":
                continue
            try:
                fname = secure_filename(f.filename)
                f.save(os.path.join(save_dir, fname))
            except Exception as e:
                warning_msg = f"Gagal upload {f.filename}: {e}"
                break
        else:
            return redirect(url_for("index"))

    people = {}
    for person in os.listdir(reference_folder):
        person_dir = os.path.join(reference_folder, person)
        if not os.path.isdir(person_dir):
            continue
        people[person] = os.listdir(person_dir)
    input_files = [f for f in os.listdir(input_folder) if is_image(f)]
    return render_template(
        "index.html",
        warning=warning_msg,
        people=people,
        input_files=input_files,
        model_list=MODEL_LIST,
        selected_model=MODEL,
        blur_threshold=BLUR_THRESHOLD
    )

@app.route("/upload_input", methods=["POST"])
def upload_input():
    files = request.files.getlist("files")
    for f in files:
        try:
            fname = secure_filename(f.filename)
            f.save(os.path.join(input_folder, fname))
        except Exception as e:
            logging.error(f"Upload error: {e}")
            return f"Gagal upload {f.filename}: {e}", 500
    return redirect(url_for("index"))

@app.route("/upload_ref_camera", methods=["POST"])
def upload_ref_camera():
    data = request.get_json()
    img_data = data.get("image")
    person_name = data.get("person_name")
    if not img_data or not person_name:
        return jsonify({"status": "error", "msg": "Data tidak lengkap"})
    try:
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        save_dir = os.path.join(reference_folder, secure_filename(person_name))
        os.makedirs(save_dir, exist_ok=True)
        fname = f"camera_{int(time.time()*1000)}.jpg"
        with open(os.path.join(save_dir, fname), "wb") as f:
            f.write(img_bytes)
        return jsonify({"status": "ok"})
    except Exception as e:
        logging.error(f"Camera ref upload error: {e}")
        return jsonify({"status": "error", "msg": str(e)})

@app.route("/process", methods=["POST"])
def process():
    blur_threshold = int(request.form.get("blur_threshold", BLUR_THRESHOLD))
    model = request.form.get("model", MODEL)
    try:
        counts, per_person, sim_threshold = run_sorting(model=model, blur_threshold=blur_threshold)
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return render_template("result.html", error=f"Gagal proses: {e}")

    # --- Mapping file input ke hasil sortir (matched) ---
    input_files_list = [f for f in os.listdir(input_folder) if is_image(f)]
    input_map = {}
    for f in input_files_list:
        base, ext = os.path.splitext(secure_filename(f))
        input_map[base] = f

    review_data = []
    for person in per_person.keys():
        person_dir = os.path.join(output_folder, person)
        if os.path.exists(person_dir):
            for f in os.listdir(person_dir):
                # Ambil nama file input asli berdasarkan prefix
                base_out, ext_out = os.path.splitext(f)
                base_in = base_out.split("_")[0]  # Ambil prefix sebelum _timestamp
                file_input = input_map.get(base_in, f)
                review_data.append({
                    "file": file_input,
                    "status": "matched",
                    "matches": [{"person": person, "similarity": sim_threshold}]
                })

    lainnya_dir = os.path.join(output_folder, "lainnya")
    if os.path.exists(lainnya_dir):
        for f in os.listdir(lainnya_dir):
            review_data.append({"file": f, "status": "lainnya", "matches": []})

    if os.path.exists(output_blur):
        for f in os.listdir(output_blur):
            review_data.append({"file": f, "status": "blur", "matches": []})

    zip_path = "hasil_sortir.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(output_folder):
            for file in files:
                path = os.path.join(root, file)
                zipf.write(path, os.path.relpath(path, output_folder))

    return render_template(
        "result.html",
        counts=counts,
        per_person=per_person,
        sim_threshold=sim_threshold,
        selected_model=model,
        blur_threshold=blur_threshold,
        review_data=review_data
    )

@app.route("/download")
def download():
    return send_file("hasil_sortir.zip", as_attachment=True)

@app.route("/download_orang")
def download_orang():
    zip_path = "hasil_orang.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for folder in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder)
            if folder == "lainnya" or not os.path.isdir(folder_path):
                continue
            for root, _, files in os.walk(folder_path):
                for file in files:
                    path = os.path.join(root, file)
                    zipf.write(path, os.path.relpath(path, output_folder))
    return send_file(zip_path, as_attachment=True)

@app.route("/download_lainnya")
def download_lainnya():
    zip_path = "hasil_lainnya.zip"
    lainnya_dir = os.path.join(output_folder, "lainnya")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if os.path.exists(lainnya_dir):
            for root, _, files in os.walk(lainnya_dir):
                for file in files:
                    path = os.path.join(root, file)
                    zipf.write(path, os.path.relpath(path, output_folder))
    return send_file(zip_path, as_attachment=True)

@app.route("/delete_ref/<person>/<filename>", methods=["POST"])
def delete_ref(person, filename):
    dir_path = os.path.join(reference_folder, secure_filename(person))
    file_path = os.path.join(dir_path, secure_filename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        if os.path.isdir(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
    return redirect(url_for("index"))

@app.route("/foto_referensi/<person>/<filename>")
def serve_ref(person, filename):
    return send_from_directory(os.path.join(reference_folder, secure_filename(person)), secure_filename(filename))

@app.route("/foto_input/<filename>")
def serve_input(filename):
    return send_from_directory(input_folder, secure_filename(filename))

@app.route("/delete_input/<filename>", methods=["POST"])
def delete_input(filename):
    file_path = os.path.join(input_folder, secure_filename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for("index"))

@app.route("/status")
def status():
    return jsonify({"status": "processing"})

if __name__ == "__main__":
    print("🚀 Starting Flask app at http://127.0.0.1:8080 ...")
    app.run(debug=True, host="0.0.0.0", port=8080)
def serve_input(filename):
    return send_from_directory(input_folder, secure_filename(filename))

@app.route("/delete_input/<filename>", methods=["POST"])
def delete_input(filename):
    file_path = os.path.join(input_folder, secure_filename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for("index"))

@app.route("/status")
def status():
    return jsonify({"status": "processing"})

if __name__ == "__main__":
    print("🚀 Starting Flask app at http://127.0.0.1:8080 ...")
    app.run(debug=True, host="0.0.0.0", port=8080)
