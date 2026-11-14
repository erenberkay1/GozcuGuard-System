<div align="center">

# 🛡️ GozcuGuard-System

**An intelligent security camera and computer vision monitoring system built with modern software architecture.**  
*Güvenlik kameralarından alınan görüntüleri modern görüntü işleme teknolojileriyle analiz eden, şüpheli durumları takip etmeyi ve güvenliği artırmayı hedefleyen akıllı yazılım sistemi.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

## 📋 Table of Contents / İçindekiler
- [About the Project / Proje Hakkında](#-about-the-project--proje-hakkında)
- [Key Features / Temel Özellikler](#-key-features--temel-özellikler)
- [Screenshots / Ekran Görüntüleri](#-screenshots--ekran-görüntüleri)
- [Installation & Running / Kurulum ve Çalıştırma](#-installation--running--kurulum-ve-çalıştırma)
- [License / Lisans](#-license--lisans)

---

## 🎯 About the Project / Proje Hakkında

**EN:** GozcuGuard-System is a robust desktop application designed for real-time video monitoring, threat detection, and privacy protection using computer vision algorithms. 

**TR:** GozcuGuard-System; bilgisayarlı görü algoritmaları kullanarak gerçek zamanlı video izleme, nesne/durum tespiti ve gizlilik koruması sağlamak amacıyla geliştirilmiş kapsamlı bir masaüstü uygulamasıdır.

---

## 🚀 Key Features / Temel Özellikler

* **🎥 Multi-Source Camera Support / Çoklu Kamera Desteği:** Live camera streams and recorded video file playback. (Canlı kamera akışları ve kaydedilmiş video dosyalarını oynatma desteği.)
* **🛡️ Privacy & Protection / Gizlilik ve Koruma:** Integrated privacy blur mechanisms. (Entegre gizlilik bulanıklaştırma mekanizmaları.)
* **⚙️ Adjustable Performance / Ayarlanabilir Performans:** Quality and speed optimization modes. (Kalite ve hız optimizasyon modları.)
* **🚨 Emergency Control / Acil Durum Kontrolü:** Quick panic key response system. (Hızlı acil durum panik tuşu tepki sistemi.)

---

## 📷 Screenshots / Ekran Görüntüleri

| Main Interface (Dashboard) / Ana Arayüz | Video Detection Module / Video Tespit Modülü |
| :---: | :---: |
| ![Main Interface](https://github.com/user-attachments/assets/c83e9d30-5244-4859-9ba6-5e3e92ed3bf9) | ![Detection Module](https://github.com/user-attachments/assets/f26be277-746b-4589-9bc5-7bee1446eb84) |

---
### 0. Prerequisites / Ön Gereksinimler (Virtual Camera / Sanal Kamera)
To use the virtual camera output in Zoom, Google Meet, or Microsoft Teams, **OBS Studio** must be installed on your system:
Zoom, Google Meet veya Microsoft Teams gibi uygulamalarda sanal kamera çıktısını kullanabilmek için bilgisayarınızda **OBS Studio** kurulu olmalıdır:

1. Download and install [OBS Studio](https://obsproject.com/). / [OBS Studio](https://obsproject.com/) indirin ve kurun.
2. Open OBS Studio once after installation to initialize the Virtual Camera driver, then you can close it. / Kurulumdan sonra sanal kamera sürücüsünün aktifleşmesi için OBS Studio'yu bir kez açıp kapatın.
   

## 🛠️ Installation & Running / Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için **CMD (Komut İstemi)** veya **PowerShell** uygulamasını açın ve sırasıyla aşağıdaki komutları yazın:

### 1. Depoyu Klonlayın (Clone the Repository)
```bash
git clone [https://github.com/erenberkay1/GozcuGuard-System.git](https://github.com/erenberkay1/GozcuGuard-System.git)
cd GozcuGuard-System
pip install -r requirements.txt
python run.py
