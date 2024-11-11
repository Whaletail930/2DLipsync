import tkinter as tk
from pathlib import Path

from LipSyncAnimator import LipSyncAnimator

VISEME_FOLDER = Path(__file__).resolve().parent.parent / "DATA/viseme_images"

image_mapping = {
    "A": f"{VISEME_FOLDER}/A.png",
    "B": f"{VISEME_FOLDER}/B.png",
    "C": f"{VISEME_FOLDER}/C.png",
    "D": f"{VISEME_FOLDER}/D.png",
    "E": f"{VISEME_FOLDER}/E.png",
    "F": f"{VISEME_FOLDER}/F.png",
    "G": f"{VISEME_FOLDER}/G.png",
    "H": f"{VISEME_FOLDER}/H.png",
    "X": f"{VISEME_FOLDER}/X.png",
}


root = tk.Tk()
app = LipSyncAnimator(root, image_mapping)
root.mainloop()
