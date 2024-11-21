import tkinter as tk
from PIL import Image, ImageTk
import os
import threading
import queue

from mfcc_extractor_lib import setup_logger
from lipsync_pytorch import run_lipsync

logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])


class LipSyncAnimator:
    def __init__(self, root, image_mapping):
        self.root = root
        self.root.title("Live Lipsync")
        self.root.geometry("500x600")  # Adjusted height to accommodate slider

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack(expand=True)

        # Slider for dB threshold
        self.slider_label = tk.Label(root, text="Set dB Threshold")
        self.slider_label.pack(pady=5)
        self.db_slider = tk.Scale(
            root, from_=-80, to=0, resolution=1, orient="horizontal", command=self.update_threshold
        )
        self.db_slider.set(-35)  # Default value
        self.db_slider.pack(pady=10)

        # Start/Stop button
        self.start_stop_button = tk.Button(root, text="Start", command=self.toggle_sequence)
        self.start_stop_button.pack(pady=10)

        # Image mapping and generator control
        self.image_mapping = image_mapping
        self.current_viseme = None
        self.is_running = False
        self.generator_thread = None
        self.queue = queue.Queue()
        self.db_threshold = -35  # Default threshold value

    def update_threshold(self, value):
        """Update the dB threshold based on the slider."""
        self.db_threshold = int(value)
        logger.info(f"dB Threshold updated to: {self.db_threshold}")

    def toggle_sequence(self):
        if self.is_running:
            self.is_running = False
            self.start_stop_button.config(text="Start")
            if self.generator_thread:
                self.generator_thread.join()
        else:
            self.is_running = True
            self.start_stop_button.config(text="Stop")
            self.generator_thread = threading.Thread(target=self.run_generator)
            self.generator_thread.start()
            self.update_image()

    def run_generator(self):
        for viseme in run_lipsync(db_threshold=self.db_threshold):  # Pass updated db_threshold to run_lipsync
            if not self.is_running:
                break
            self.queue.put(viseme)

    def update_image(self):
        if not self.is_running:
            return

        try:
            while not self.queue.empty():
                new_viseme = self.queue.get_nowait()

                if new_viseme != self.current_viseme:
                    self.current_viseme = new_viseme
                    filename = self.image_mapping.get(new_viseme)

                    if filename and os.path.exists(filename):
                        img = Image.open(filename)
                        img = img.resize((400, 400), Image.LANCZOS)
                        img_tk = ImageTk.PhotoImage(img)
                        self.image_label.config(image=img_tk)
                        self.image_label.image = img_tk
                    else:
                        logger.error(f"Image not found for viseme: {new_viseme}")

        except queue.Empty:
            pass

        self.root.after(10, self.update_image)
