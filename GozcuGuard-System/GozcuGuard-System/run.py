"""Başlatıcı:  python run.py            -> arayüz
               python run.py cli --source test.mp4 --no-virtualcam --frames 100"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from guvenli_kamera.main import main

if __name__ == "__main__":
    main()
