import tkinter as tk

import cv2
from PIL import Image, ImageTk

root = tk.Tk()

# Load image with OpenCV
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to PIL format
img_pil = Image.fromarray(img)
img_tk = ImageTk.PhotoImage(img_pil)

# Create GUI
panel = tk.Label(root, image=img_tk)
panel.pack()


def button_callback():
    print("Button pressed")


button = tk.Button(root, text="Process", command=button_callback)
button.pack()

root.mainloop()
