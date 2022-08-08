# -*- coding: utf-8 -*-
"""
Created on 2017/5/21

@author: Naiive
选择文件
"""
from tkinter import filedialog

from PIL import Image
from tkinter import *

root = Tk()
root.title("基于深度学习的图像识别")
root.geometry('640x360')
Frame1 = Frame(root, height = 200,width = 400)
Frame2 = Frame(root, height = 200,width = 400)
Frame1.pack()
Frame2.pack(side = BOTTOM)

def select():
    global filename
    filename = filedialog.askopenfilename(filetypes = [('JPG', '.jpg'), ('PNG', '.png'), ('BMP', '.bmp')])
    print (filename)

def show():
    bm = PhotoImage(filename)
    label = Label(Frame1, image = bm)
    label.pack()

b5 = Button(Frame2, text = "选择图片", command = select,width=30, height=2, bd = 5)


b1 = Button(Frame1, text = "预测", command = show,width=30, height=2, bd = 5)
b1.pack()
b5.pack()


root.mainloop()