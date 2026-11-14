"""MediaPipe model dosyalarını assets/models klasörüne indirir (kıyafet modu ve arka plan için)."""
import os
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models/"
FILES = {
    "pose_landmarker_lite.task": BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "selfie_segmenter.tflite": BASE + "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models")
os.makedirs(dest, exist_ok=True)
for name, url in FILES.items():
    path = os.path.join(dest, name)
    if os.path.exists(path):
        print("var:", name)
        continue
    try:
        print("indiriliyor:", name)
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        print("HATA (%s): %s\nBağlantıyı elle kontrol edin: %s" % (name, exc, url))
