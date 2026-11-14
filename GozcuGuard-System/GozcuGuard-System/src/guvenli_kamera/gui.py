"""Simple interface (customtkinter)."""
from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image
import numpy as np

from . import CREDIT, __version__, credit_ok
from .prefs import load_prefs, save_prefs
from .pipeline import Pipeline
from .camera import FakeCamera
from .outfit.overlay import OverlayProvider, OVERLAY_ADJUST
from .outfit.garment import GarmentManager, make_template, TEMPLATE_SHOULDERS, TEMPLATE_NAMES
from .outfit.base import OutfitProvider
from .outfit.none import NoOpOutfitProvider

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 840


class SafeCameraGUI:
    """Main interface."""

    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("SafeCamera v%s  -  %s" % (__version__, CREDIT))
        screen_h = self.root.winfo_screenheight()
        height = max(560, min(DEFAULT_WINDOW_HEIGHT, int(screen_h) - 90))
        self.root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{height}")
        self.root.minsize(760, 520)
        self.root.resizable(True, True)
        self._fullscreen = False
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self._set_fullscreen(False))

        self._pipeline: Optional[Pipeline] = None
        self._preview_running = True
        self._panik_active = False
        self._preview_after_id: Optional[str] = None

        self.garment_manager = GarmentManager()
        self.current_outfit_provider: Optional[OutfitProvider] = None
        self._outfit_lock = threading.Lock()

        self._prefs = load_prefs()
        self.male_breast_var = ctk.BooleanVar(value=self._prefs["male_chest"])
        self.nudity_blur_var = ctk.BooleanVar(value=self._prefs["nudity_blur"])
        self.background_blur_var = ctk.BooleanVar(value=self._prefs["background_blur"])
        self._garment_color = (40, 70, 140)
        self._using_template = True
        self._current_template = TEMPLATE_NAMES[0]

        self._build_ui()
        self.start_btn.configure(state="normal")
        self.status_var.set("Status: Ready. Select camera and press 'Start'")
        self._preview_running = True
        self._update_preview()
        if not credit_ok():
            self.start_btn.configure(state="disabled")
            self.status_var.set("This copy has been modified. Please get the original SafeCamera by Eren Berkay Gun.")
        self.root.after(400, self._maybe_show_welcome)

    def _build_ui(self):
        self._bottom_bar = ctk.CTkFrame(self.root)
        self._bottom_bar.pack(side="bottom", fill="x", padx=10, pady=(0, 8))

        main_frame = ctk.CTkScrollableFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=(10, 4))

        self.video_label = ctk.CTkLabel(main_frame, text="Loading...",
                                        bg_color="black", fg_color="black", text_color="white")
        self.video_label.pack(fill="x", expand=False, pady=(0, 10))

        ctrl_frame = ctk.CTkFrame(main_frame)
        ctrl_frame.pack(fill="x", pady=(0, 10))

        left_frame = ctk.CTkFrame(ctrl_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))

        ctk.CTkLabel(left_frame, text="Camera:").pack(anchor="w")
        self.camera_combo = ctk.CTkOptionMenu(left_frame, values=["Real Camera", "Test Video"],
                                             command=self._on_camera_mode_changed)
        self.camera_combo.set("Real Camera")
        self.camera_combo.pack(fill="x", pady=5)

        ctk.CTkLabel(left_frame, text="Source:").pack(anchor="w")
        self.source_entry = ctk.CTkEntry(left_frame, width=150)
        self.source_entry.insert(0, "0")
        self.source_entry.pack(fill="x", pady=5)
        ctk.CTkLabel(left_frame, text="Quality (speed):").pack(anchor="w", pady=(8, 0))
        self.quality_menu = ctk.CTkOptionMenu(left_frame, values=list(self.QUALITY_MAP.keys()))
        self.quality_menu.set("Balanced (960x540)")
        self.quality_menu.pack(fill="x", pady=5)

        mid_frame = ctk.CTkFrame(ctrl_frame)
        mid_frame.pack(side="left", fill="y", padx=10, expand=True)

        ctk.CTkLabel(mid_frame, text="Outfit Mode:").pack(anchor="w", pady=(2, 0))
        self.outfit_menu = ctk.CTkOptionMenu(mid_frame, values=["Off", "Light", "Realistic"],
                                             command=self._on_outfit_mode_changed)
        self.outfit_menu.set("Off")
        self.outfit_menu.pack(fill="x", pady=2)

        self.load_garment_btn = ctk.CTkButton(mid_frame, text="Load Garment",
                                               command=self._on_load_garment)
        self.load_garment_btn.pack(fill="x", pady=1)

        ctk.CTkLabel(mid_frame, text="Template:").pack(anchor="w", pady=(6, 0))
        self.template_menu = ctk.CTkOptionMenu(mid_frame,
                                               values=list(TEMPLATE_NAMES),
                                               command=self._on_template_changed)
        self.template_menu.set(TEMPLATE_NAMES[0])
        self.template_menu.pack(fill="x", pady=1)

        self.color_btn = ctk.CTkButton(mid_frame, text="Choose Color",
                                       command=self._on_color_choose,
                                       width=80)
        self.color_btn.pack(fill="x", pady=1)

        self.preview_label = ctk.CTkLabel(mid_frame, text="", width=80, height=80,
                                         fg_color=("gray70", "gray30"))
        self.preview_label.pack(fill="x", pady=2)
        ctk.CTkLabel(mid_frame, text="80x80 preview", font=("", 8)).pack(anchor="w", padx=4)

        ctk.CTkLabel(mid_frame, text="Outfit Adjust (live):").pack(anchor="w", pady=(6, 0))
        self.scale_slider = ctk.CTkSlider(mid_frame, from_=0.6, to=1.6, number_of_steps=40,
                                          command=self._on_outfit_adjust)
        self.scale_slider.set(1.0)
        ctk.CTkLabel(mid_frame, text="Size").pack(anchor="w")
        self.scale_slider.pack(fill="x")
        self.dy_slider = ctk.CTkSlider(mid_frame, from_=-0.4, to=0.4, number_of_steps=40,
                                       command=self._on_outfit_adjust)
        self.dy_slider.set(0.0)
        ctk.CTkLabel(mid_frame, text="Up  <->  Down").pack(anchor="w")
        self.dy_slider.pack(fill="x")
        self.dx_slider = ctk.CTkSlider(mid_frame, from_=-0.4, to=0.4, number_of_steps=40,
                                       command=self._on_outfit_adjust)
        self.dx_slider.set(0.0)
        ctk.CTkLabel(mid_frame, text="Left  <->  Right").pack(anchor="w")
        self.dx_slider.pack(fill="x")
        self.reset_outfit_btn = ctk.CTkButton(mid_frame, text="Reset Adjust", height=22,
                      command=self._reset_outfit_adjust).pack(fill="x", pady=(4, 2))


        right_frame = ctk.CTkFrame(ctrl_frame)
        right_frame.pack(side="right", fill="y", expand=True)

        ctk.CTkLabel(right_frame, text="Privacy & Protection", font=("", 13, "bold")).pack(anchor="w", pady=(2, 4))

        # Blur system 1: nudity blur (opt-in)
        self.nudity_switch = ctk.CTkSwitch(right_frame, text="Nudity blur", variable=self.nudity_blur_var,
                                           command=self._on_nudity_toggle)
        self.nudity_switch.pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(right_frame, text="Detects nudity and blurs it. Not 100% reliable.",
                     font=("", 9), text_color="gray70", wraplength=230, justify="left").pack(anchor="w", padx=(40, 0))
        self.male_breast_check = ctk.CTkCheckBox(right_frame, text="Also blur bare male chest",
                                                 variable=self.male_breast_var,
                                                 command=self._on_male_chest_toggle)
        self.male_breast_check.pack(anchor="w", padx=(40, 0), pady=(4, 6))

        ctk.CTkLabel(right_frame, text="Safety Policy:").pack(anchor="w")
        self.policy_menu = ctk.CTkOptionMenu(right_frame, values=["Regional", "Strict", "Cooldown", "Protection Screen"])
        self.policy_menu.set("Regional")
        self.policy_menu.pack(fill="x", pady=5)

        # Blur system 2: background blur (opt-in)
        self.background_switch = ctk.CTkSwitch(right_frame, text="Background blur",
                                               variable=self.background_blur_var,
                                               command=self._on_background_toggle)
        self.background_switch.pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(right_frame, text="Keeps you sharp and blurs everything behind you.",
                     font=("", 9), text_color="gray70", wraplength=230, justify="left").pack(anchor="w", padx=(40, 0))
        self._sync_nudity_widgets()

        btn_frame = ctk.CTkFrame(self._bottom_bar)
        btn_frame.pack(fill="x", pady=10)

        self.start_btn = ctk.CTkButton(btn_frame, text="Start", command=self._toggle_start,
                                       state="disabled")
        self.start_btn.pack(side="left", padx=5)

        self.panik_btn = ctk.CTkButton(btn_frame, text="Panic Key", fg_color="red",
                                       hover_color="darkred", command=self._trigger_panik)
        self.panik_btn.pack(side="left", padx=5)

        self.status_var = ctk.StringVar(value="Status: Loading...")
        status_bar = ctk.CTkLabel(self._bottom_bar, textvariable=self.status_var,
                                  anchor="w", height=2, fg_color="gray20", text_color="white")
        status_bar.pack(fill="x", pady=(5, 0))

        self._tip_label = ctk.CTkLabel(self._bottom_bar, text="Tip: Transparent PNG background gives best results. Plain white background product photos also work.",
                                       font=("", 9), text_color="orange")
        self._tip_label.pack(anchor="w", pady=(0, 5))
        # Credit / imza
        self._credit_label = ctk.CTkLabel(self._bottom_bar, text=CREDIT,
                                          font=("", 10, "bold"), text_color="gray75")
        self._credit_label.pack(anchor="w", pady=(0, 5))

    def _set_fullscreen(self, on: bool) -> None:
        self._fullscreen = bool(on)
        try:
            self.root.attributes("-fullscreen", self._fullscreen)
        except Exception as exc:
            logger.error("Fullscreen error: %s", exc)

    def _toggle_fullscreen(self, _event=None) -> None:
        self._set_fullscreen(not self._fullscreen)

    POLICY_MAP = {"Regional": "regional", "Strict": "solid",
                  "Cooldown": "cooldown", "Protection Screen": "overlay"}

    QUALITY_MAP = {"Fast (640x480)": (640, 480), "Balanced (960x540)": (960, 540), "HD (1280x720)": (1280, 720)}

    def _build_pipeline(self) -> Pipeline:
        fake = self.camera_combo.get() == "Test Video"
        src = self.source_entry.get().strip()
        if not fake and not src.isdigit():
            src = "0"
        provider = self.current_outfit_provider if self.current_outfit_provider is not None else NoOpOutfitProvider({"mode": "off"})

        pipe = Pipeline(camera_source=src, use_fake_camera=fake, fake_source=src,
                        safety_policy=self.POLICY_MAP.get(self.policy_menu.get(), "regional"),
                        outfit_provider=provider,
                        enable_panik=False, use_virtual_camera=True,
                        male_breast_exposed=bool(self.male_breast_var.get()),
                        enable_nudity_blur=bool(self.nudity_blur_var.get()),
                        enable_background_blur=bool(self.background_blur_var.get()),
                        resolution=self.QUALITY_MAP.get(self.quality_menu.get(), (960, 540)))
        return pipe

    def _on_camera_mode_changed(self, choice):
        self.source_entry.delete(0, "end")
        self.source_entry.insert(0, "test.mp4" if choice == "Test Video" else "0")

    def _on_outfit_mode_changed(self, choice: str) -> None:
        logger.info("Outfit mode changed: %s", choice)
        if choice == "Realistic":
            self.outfit_menu.set("Light")
            self.status_var.set("Realistic (cloud) mode not ready yet, using Light mode")
            choice = "Light"

        if choice == "Off":
            provider = NoOpOutfitProvider({"mode": "off"})
        else:
            with self._outfit_lock:
                cur = self.current_outfit_provider
            if isinstance(cur, OverlayProvider) and cur._ready:
                provider = cur
            else:
                self._on_template_changed(self.template_menu.get())
                return

        with self._outfit_lock:
            self.current_outfit_provider = provider
            if self._pipeline is not None:
                self._pipeline.set_outfit(provider)

        self.status_var.set(f"Status: Outfit mode set to {choice}")

    def _on_outfit_adjust(self, _value=None) -> None:
        OVERLAY_ADJUST["scale"] = float(self.scale_slider.get())
        OVERLAY_ADJUST["dy"] = float(self.dy_slider.get())
        OVERLAY_ADJUST["dx"] = float(self.dx_slider.get())

    def _reset_outfit_adjust(self) -> None:
        self.scale_slider.set(1.0)
        self.dy_slider.set(0.0)
        self.dx_slider.set(0.0)
        self._on_outfit_adjust()

    def _on_template_changed(self, choice: str) -> None:
        arr = make_template(choice, self._garment_color)
        prov = OverlayProvider({"mode": "overlay"})
        if not prov.set_garment_array(arr, shoulders=TEMPLATE_SHOULDERS):
            self.status_var.set("Error: Template could not be created")
            return
        self._using_template = True
        self._current_template = choice
        with self._outfit_lock:
            self.current_outfit_provider = prov
            if self._pipeline is not None:
                self._pipeline.set_outfit(prov)
        self.outfit_menu.set("Light")
        self.status_var.set(f"Status: Template: {choice}. Wearer face should be clear in camera")

    @staticmethod
    def _resolve_asset_path(filename: str) -> str | None:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", filename),
            os.path.join("assets", filename),
        ]
        for p in candidates:
            if os.path.exists(p):
                return os.path.abspath(p)
        return None

    def _on_load_garment(self) -> None:
        import tkinter.filedialog as fd
        fpath = fd.askopenfilename(title="Select Garment Image",
                                   filetypes=[("Image files", "*.png *.jpg *.jpeg"),
                                              ("All files", "*.*")])
        if not fpath:
            return
        provider = OverlayProvider({"mode": "overlay"})
        if not provider.load_garment(fpath):
            self.status_var.set("Error: Garment image could not be loaded")
            return
        self._using_template = False
        with self._outfit_lock:
            self.current_outfit_provider = provider
            if self._pipeline is not None:  # apply immediately while the camera is running
                self._pipeline.set_outfit(provider)
        self.outfit_menu.set("Light")
        self._update_garment_preview(fpath)
        self.status_var.set("Status: Garment loaded. Use only images you have the right to use. Fine-tune with the sliders")

    def _on_color_choose(self) -> None:
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Choose Garment Color")
        if not (color and color[0]):
            return
        self._garment_color = tuple(int(c) for c in color[0])
        if self._using_template:
            self._on_template_changed(self._current_template)
        else:
            self.status_var.set("Color selection applies only to templates")

    def _save_prefs(self) -> None:
        self._prefs.update({"nudity_blur": bool(self.nudity_blur_var.get()),
                            "background_blur": bool(self.background_blur_var.get()),
                            "male_chest": bool(self.male_breast_var.get())})
        save_prefs(self._prefs)

    def _sync_nudity_widgets(self) -> None:
        """Sub-options only make sense while the nudity blur is on."""
        state = "normal" if self.nudity_blur_var.get() else "disabled"
        for w in (self.male_breast_check, self.policy_menu):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _on_nudity_toggle(self) -> None:
        on = bool(self.nudity_blur_var.get())
        if self._pipeline is not None:
            self._pipeline.set_nudity_blur_enabled(on)
        self._sync_nudity_widgets()
        self._save_prefs()
        self.status_var.set("Status: Nudity blur %s" % ("ON" if on else "OFF"))

    def _on_background_toggle(self) -> None:
        on = bool(self.background_blur_var.get())
        if self._pipeline is not None:
            self._pipeline.set_background_blur_enabled(on)
        self._save_prefs()
        self.status_var.set("Status: Background blur %s" % ("ON" if on else "OFF"))

    def _on_male_chest_toggle(self) -> None:
        on = bool(self.male_breast_var.get())
        if self._pipeline is not None:
            self._pipeline.set_include_male_breast(on)
        self._save_prefs()
        self.status_var.set("Status: Bare male chest blur %s" % ("ON" if on else "OFF"))

    def _maybe_show_welcome(self) -> None:
        """First start: ask the user which blur systems to enable."""
        if self._prefs.get("onboarding_done"):
            return
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Welcome to SafeCamera")
        dlg.geometry("460x340")
        dlg.resizable(False, False)
        try:
            dlg.transient(self.root)
            dlg.grab_set()
        except Exception:
            pass
        ctk.CTkLabel(dlg, text="Choose your protections", font=("", 16, "bold")).pack(pady=(16, 4))
        ctk.CTkLabel(dlg, text="Both are optional. You can change them any time in the main window.",
                     wraplength=400, text_color="gray70").pack(pady=(0, 10))
        n_var = ctk.BooleanVar(value=True)
        b_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(dlg, text="Nudity blur (recommended)", variable=n_var).pack(anchor="w", padx=30, pady=(6, 0))
        ctk.CTkLabel(dlg, text="Detects nudity on camera and blurs it automatically.",
                     font=("", 10), text_color="gray70").pack(anchor="w", padx=56)
        ctk.CTkCheckBox(dlg, text="Background blur", variable=b_var).pack(anchor="w", padx=30, pady=(12, 0))
        ctk.CTkLabel(dlg, text="Blurs everything except you.",
                     font=("", 10), text_color="gray70").pack(anchor="w", padx=56)

        def _continue():
            self.nudity_blur_var.set(bool(n_var.get()))
            self.background_blur_var.set(bool(b_var.get()))
            self._prefs["onboarding_done"] = True
            self._save_prefs()
            self._sync_nudity_widgets()
            try:
                dlg.destroy()
            except Exception:
                pass
            self.root.after(300, self._show_obs_help)  # first-run: explain the virtual camera driver

        ctk.CTkButton(dlg, text="Continue", command=_continue).pack(pady=22)
        ctk.CTkLabel(dlg, text=CREDIT, font=("", 9), text_color="gray60").pack(side="bottom", pady=8)
        dlg.protocol("WM_DELETE_WINDOW", _continue)

    OBS_URL = "https://obsproject.com/download"

    def _show_obs_help(self, reason: str = "") -> None:
        """Explain that a virtual camera driver (OBS Virtual Camera) is needed for Zoom/Meet/Teams."""
        if getattr(self, "_obs_help_open", False):
            return
        self._obs_help_open = True
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Virtual camera needed")
        dlg.geometry("540x430")
        dlg.resizable(False, False)
        try:
            dlg.transient(self.root)
        except Exception:
            pass

        def _close():
            self._obs_help_open = False
            try:
                dlg.destroy()
            except Exception:
                pass

        ctk.CTkLabel(dlg, text="Virtual camera driver needed", font=("", 16, "bold")).pack(pady=(16, 6))
        if reason:
            ctk.CTkLabel(dlg, text=reason, wraplength=490, text_color="#ff8080").pack(pady=(0, 6))
        body = ("To use SafeCamera inside Zoom, Google Meet or Microsoft Teams, Windows needs a virtual "
                "camera. The free OBS Studio installs one (\"OBS Virtual Camera\").\n\n"
                "1. Install OBS Studio and open it once.\n"
                "2. Click \"Start Virtual Camera\", then \"Stop Virtual Camera\", and close OBS completely.\n"
                "3. Start SafeCamera and press Start BEFORE you open Zoom.\n"
                "4. In Zoom/Meet/Teams choose \"OBS Virtual Camera\" as the camera.\n\n"
                "SafeCamera is an independent project and is not affiliated with or endorsed by the OBS "
                "Project, Zoom, Google or Microsoft. All trademarks belong to their owners.")
        ctk.CTkLabel(dlg, text=body, wraplength=490, justify="left").pack(padx=24, pady=6)
        ctk.CTkButton(dlg, text="Open OBS download page",
                      command=lambda: webbrowser.open(self.OBS_URL)).pack(pady=(10, 4))
        ctk.CTkButton(dlg, text="Close", fg_color="gray30", command=_close).pack(pady=4)
        dlg.protocol("WM_DELETE_WINDOW", _close)

    def _update_outfit_status(self, provider: OutfitProvider) -> None:
        try:
            status = provider.get_status()
        except Exception:
            status = "Unknown"
        current = self.outfit_menu.get().split(" (")[0] if " (" in self.outfit_menu.get() else self.outfit_menu.get()
        self.outfit_menu.set(current)
        self.status_var.set(f"Status: Outfit active - {status}")

    def _update_garment_preview(self, path: str) -> None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (80, 80), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(img_rgb)
            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(80, 80))
            self.preview_label.configure(image=ctk_img, text="")
            self.preview_label.image = ctk_img
        except Exception as exc:
            logger.error("Preview update error: %s", exc)

    def _toggle_start(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self.start_btn.configure(text="Start")
            self.status_var.set("Status: Stopped")
            self.video_label.configure(image="", text="Camera closed")
            return
        pipe = self._build_pipeline()
        self.status_var.set("Status: Opening camera...")
        self.root.update_idletasks()
        if not pipe.start():
            self.status_var.set("Status: " + (pipe.error or "Could not start"))
            return
        pipe.set_panic(self._panik_active)
        with self._outfit_lock:
            pipe.set_outfit(self.current_outfit_provider or NoOpOutfitProvider({"mode": "off"}))
        self._pipeline = pipe
        self.start_btn.configure(text="Stop")
        if not getattr(pipe.virtual, "is_opened", False):
            self._show_obs_help(getattr(pipe.virtual, "error", None) or "Virtual camera could not be opened.")

    def _trigger_panik(self):
        self._panik_active = not self._panik_active
        status = "ACTIVE" if self._panik_active else "INACTIVE"
        self.status_var.set(f"Status: Panic {status}")
        if self._pipeline is not None:
            self._pipeline.set_panic(self._panik_active)

    def _update_preview(self):
        if not self._preview_running:
            return
        try:
            if self.root is None or not self.root.winfo_exists():
                self._preview_running = False
                return
        except Exception:
            self._preview_running = False
            return

        pipe = self._pipeline
        if pipe is not None:
            frame = pipe.get_last_frame()
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    scale = min(1.0, 560.0 / w)
                    size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    img = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
                    self.video_label.configure(image=img, text="")
                    self.video_label.image = img
                except Exception as exc:
                    logger.error("Preview error: %s", exc)
            st = pipe.status()
            if not st["nudity_blur"]:
                nud = "OFF"
            elif st["failed"]:
                nud = "FAILED (camera disabled)"
            else:
                nud = "ON" if st["detector"] else "loading..."
            bgs = "OFF"
            if st["background_blur"]:
                bgs = "ON (approx.)" if st["mask_mode"] == "approximate" else "ON"
            parts = ["Status: Running", "FPS %.0f (work %.0f ms, camera wait %.0f ms, %dx%d)" % (st["fps"], st["proc_ms"], st["wait_ms"], st["size"][0], st["size"][1]),
                     "Nudity blur: " + nud, "Background blur: " + bgs,
                     "Virtual camera: " + ("open" if st["virtual"] else "closed (OBS needed)")]
            if st.get("last_detection"):
                parts.append("Last detection: " + st["last_detection"])
            if st["panic"]:
                parts.append("PANIC ACTIVE")
            if st["error"]:
                parts.append(st["error"][:80])
            try:
                with self._outfit_lock:
                    prov = self.current_outfit_provider
                if prov is not None:
                    parts.append("Outfit: " + prov.get_status())
                else:
                    parts.append("Outfit: Off")
            except Exception:
                parts.append("Outfit: Unknown")
            self.status_var.set(" | ".join(parts))

        if self._preview_running and self.root.winfo_exists():
            try:
                self._preview_after_id = self.root.after(60, self._update_preview)
            except Exception:
                self._preview_running = False

    def run(self):
        self.root.mainloop()

    def cleanup(self):
        self._preview_running = False
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass


def run_gui():
    app = SafeCameraGUI()
    app.run()


if __name__ == "__main__":
    run_gui()
