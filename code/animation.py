import tkinter as tk
from pathlib import Path

from LipSyncAnimator import LipSyncAnimator

VISEME_FOLDER = Path(__file__).resolve().parent.parent / "DATA/viseme_images"

image_mapping = {
    0: f"{VISEME_FOLDER}/A.png",
    1: f"{VISEME_FOLDER}/B.png",
    2: f"{VISEME_FOLDER}/C.png",
    3: f"{VISEME_FOLDER}/D.png",
    4: f"{VISEME_FOLDER}/E.png",
    5: f"{VISEME_FOLDER}/F.png",
    6: f"{VISEME_FOLDER}/G.png",
    7: f"{VISEME_FOLDER}/H.png",
    8: f"{VISEME_FOLDER}/X.png",
}


root = tk.Tk()
app = LipSyncAnimator(root, image_mapping)
root.mainloop()
