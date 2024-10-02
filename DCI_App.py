TF_ENABLE_ONEDNN_OPTS = 0

import numpy as np
import tkinter 
from tkinter import filedialog, messagebox
import tensorflow as tf
from PIL import Image, ImageTk, UnidentifiedImageError
from rembg import remove
import cv2

d = {}

def getImage():

    global d
        
    try:
        path = filedialog.askopenfilename()
        im = Image.open(path)

        img_width, img_height = im.size

        if img_width > WIDTH or img_height > HEIGHT:

            while img_width > WIDTH or img_height > HEIGHT:
                img_width *= .99
                img_height *= .99
            im = im.resize((int(img_width), int(img_height)))
            messagebox.showinfo(title = 'Warning!', message = 'The uploaded image is larger than the canvas. It will be resized.')
        
            img = ImageTk.PhotoImage(im)
            text = image_prediction(path)   

            d['img'] = img
            d['text'] = text

            new_window()
        
        else:

            img = ImageTk.PhotoImage(im)
            text = image_prediction(path)   

            d['img'] = img
            d['text'] = text

            new_window()

    except UnidentifiedImageError:
        messagebox.showinfo(title = 'Upload Error', message = 'Image could not be read, please make sure the selected is an image file')

    except PermissionError:
        messagebox.showinfo(title = 'Upload Error', message = 'Please make sure that an image file is selected')


def predict(model, img):

    class_names = ['Bharatanatyam', 'Kathakali', 'Kuchipudi', 'Mohiniyattam']

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


def loading(image_path):
        
    class_names = ['Bharatanatyam', 'Kathakali', 'Kuchipudi', 'Mohiniyattam']

    image = image_path
    image = image.resize((224, 224))

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    model_path1 = './DCI_CNN1.h5'
    model_path2 = './DCI_VGG1.h5'

    loaded_model1 = tf.keras.models.load_model(model_path1)
    img_array = tf.expand_dims(image_array, 0)

    predictions = loaded_model1.predict(img_array)
    predicted_class1 = class_names[np.argmax(predictions[0])]
    confidence1 = round(100 * (np.max(predictions[0])), 2)

    loaded_model2 = tf.keras.models.load_model(model_path2)
    img_array = tf.expand_dims(image_array, 0)

    predictions = loaded_model2.predict(img_array)
    predicted_class2 = class_names[np.argmax(predictions[0])]
    confidence2 = round(100 * (np.max(predictions[0])), 2)

    if confidence1 > confidence2:
        return predicted_class1, confidence1, 'CNN'
    else:
        return predicted_class2, confidence2, 'VGG 16'


def preprocess(path):

    input = cv2.imread(path)
    img_processed = remove(input)
    img_processed = Image.fromarray(img_processed)

    return img_processed


def image_prediction(path):

    input_img = preprocess(path)
    results = loading(input_img)

    return 'Preprocessing technique : Background Removal\n' + 'Model : ' + results[2] + '\n' + 'Predicted : ' + results[0] + "\n Confidence : " + str(results[1]) + '%'

WIDTH, HEIGHT = 500, 500

app = tkinter.Tk()
app.geometry('%sx%s' % (500, 300))
app.title("Costume based dance classifier")

upload_label = tkinter.Label(text = "To select an image press the \n'Open' button", font = ('Arial', 15, 'bold'))
upload_label.config(fg = 'blue')
upload_label.place(relx = 0.5, rely = 0.4, anchor = 'center')


def new_window():

    global d

    WIDTH, HEIGHT = 500, 500

    app1 = tkinter.Toplevel(app)
    app1.geometry('%sx%s' % (600, 760))
    app1.title("Costume based dance classifier")
    app1.wm_attributes("-topmost", True)

    img, text = d['img'], d['text']

    canvas = tkinter.Canvas(app1, width = WIDTH, height = HEIGHT, bg = 'white')
    canvas.img = img
    canvas.create_image(WIDTH / 2, HEIGHT / 2, image = img, anchor = tkinter.CENTER)
    canvas.pack()

    result_label = tkinter.Label(app1, text = text, font = ('Arial', 15, 'bold'))
    result_label.config(fg = 'green')
    result_label.place(relx = 0.5, rely = 0.75, anchor = 'center')

    close_button = tkinter.Button(app1, 
                   text = "Close", 
                   command = app1.destroy,
                   activebackground = "blue", 
                   activeforeground = "white",
                   anchor = "center",
                   bd = 3,
                   bg = "lightgray",
                   cursor = "hand2",
                   disabledforeground = "gray",
                   fg = "black",
                   font = ("Arial", 12),
                   height = 2,
                   highlightbackground = "black",
                   highlightcolor = "green",
                   highlightthickness = 2,
                   justify = "center",
                   overrelief = "raised",
                   padx = 10,
                   pady = 5,
                   width = 15,
                   wraplength = 100)
    
    close_button.place(relx = 0.5, rely = 0.9, anchor = 'center')

    app1.mainloop()

upload_button = tkinter.Button(app, 
                   text = "Open", 
                   command = getImage,
                   activebackground = "blue", 
                   activeforeground = "white",
                   anchor = "center",
                   bd = 3,
                   bg = "lightgray",
                   cursor = "hand2",
                   disabledforeground = "gray",
                   fg = "black",
                   font = ("Arial", 12),
                   height = 2,
                   highlightbackground = "black",
                   highlightcolor = "green",
                   highlightthickness = 2,
                   justify = "center",
                   overrelief = "raised",
                   padx = 10,
                   pady = 5,
                   width = 15,
                   wraplength = 100)

upload_button.place(relx = 0.3, rely = 0.7, anchor = 'center')

close_button = tkinter.Button(app, 
                   text = "Close", 
                   command = app.destroy,
                   activebackground = "blue", 
                   activeforeground = "white",
                   anchor = "center",
                   bd = 3,
                   bg = "lightgray",
                   cursor = "hand2",
                   disabledforeground = "gray",
                   fg = "black",
                   font = ("Arial", 12),
                   height = 2,
                   highlightbackground = "black",
                   highlightcolor = "green",
                   highlightthickness = 2,
                   justify = "center",
                   overrelief = "raised",
                   padx = 10,
                   pady = 5,
                   width = 15,
                   wraplength = 100)
    
close_button.place(relx = 0.7, rely = 0.7, anchor = 'center')

app.mainloop()