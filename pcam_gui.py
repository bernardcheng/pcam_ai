from tkinter import *
from tkinter import filedialog, messagebox
import os
import cv2
from PIL import ImageTk, Image
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

IMG_WIDTH = 96
IMG_HEIGHT = 96
MODEL_FILE = 'resnet-all-freeze.h5'

# On-click functions

def browse():
    # Allow user to select a directory and store it in global var called filename
    global filename, selected
    filename = filedialog.askopenfilename(filetypes = (("jpeg files","*.jpg"),("png files","*.png*")))
    imgPath_label.config(text=filename)
    print('Selected file: ',filename)
    selected = True

    # Display selected image in imgDisplay_label
    tkimage = ImageTk.PhotoImage(Image.open(filename).resize((300,300)))
    imgDisplay_label = Label(window, image = tkimage).grid(row = 4, column = 0, sticky = W)
    imgDisplay_label.pack() # Ignore Attribute Error


def process():
    if not 'filename' in globals():
        messagebox.showerror("Processing Error", "No file detected: Please select valid file under Insert image to label.")
    else:
        global filename
        # pred_label.delete(0.0, END)
        model = load_model(os.path.join(os.getcwd(), 'model_ckpt') + '\\' + MODEL_FILE)

        new_image = image.load_img(filename, target_size=(IMG_HEIGHT,IMG_WIDTH))
        x = image.img_to_array(new_image)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)


        # check prediction
        pred = model.predict(x, batch_size = 1, verbose = 1)
        print('Prediction Score: ', pred)

        if pred[0][0] >= .5:
            pred_label.config(text=f'Prediction Result: Malignant ({pred[0][0]})')
        else:
            pred_label.config(text=f'Prediction Result: Benign ({pred[0][0]})')


def close_window():
    window.destroy()
    exit()

### Main:
window = Tk()
window.title("PCAM Classification")
window.configure(background ="white")
window.iconbitmap('sutd.ico')

### Tkinter GUI Layout

### Create Label
insertimg_label = Label(window, text = "Insert image file to label", bg = "white", fg = "black", font = "none 12 bold")
insertimg_label.grid(row = 0, column = 0, sticky = W)

### Add a browse button to search for file
Button(window, text = "Browse", width = 10, command = browse).grid(row = 3, column = 1, sticky = E)

### Create label box for file directory
imgPath_label = Label(window, bg = "white", fg = "black", font = "none 10",)
imgPath_label.grid(row = 3, column = 0, sticky = W)

### Loads selected image after browse
selected_img = Image.open("no_img_selected.png").resize((300,300))
tkimage = ImageTk.PhotoImage(selected_img)
imgDisplay_label = Label(window, image = tkimage).grid(row = 4, column = 0, sticky = W)

### Add a Predict button to run model on selected image
Button(window, text = "Predict", width = 10, command = process).grid(row = 5, column = 1, sticky = W)

### Prediction Label
Label(window, text = "Prediction", bg = "white", fg = "black", font = "none 12 bold").grid(row =6, column = 0, sticky = W)

### Create label box for prediction results
pred_label = Label(window, text = "Not Implemented Yet.", bg = "white", fg = "black", font = "none 10",)
pred_label.grid(row = 7, column = 0, sticky = W)

### Exit Label
Label(window, text = "Click to exit:", bg = "white", fg = "black", font = "none 12 bold").grid(row =14, column = 0, sticky = W)

### Add exit Button
Button(window, text = "Exit", width = 14, command = close_window).grid(row = 15, column = 0, sticky = W)

### Run main loop
window.mainloop()
