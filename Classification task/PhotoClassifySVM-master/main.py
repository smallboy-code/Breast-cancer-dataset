# -*- coding: utf-8 -*-
"""
Created on 2017/5/21

@author: Naiive
主函数
"""
from tkinter import filedialog

from PIL import Image as IMG
from PIL import ImageTk
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time
import matplotlib.pyplot as plt
from get_eigenvalue import *
from get_svm import *
from tkinter import *

root = Tk()
root.title("基于深度学习的图像识别")
root.geometry('510x280')
menubar = Menu(root)


# Frame1 = Frame(root, height = 200,width = 400)
# Frame2 = Frame(root, height = 200,width = 400)
# Frame3 = Frame(root, height = 200,width = 400)
# Frame1.pack()
# Frame2.pack()
# Frame3.pack(side = BOTTOM)

target_n = None
filename = None

canvas = Canvas(root,
                    width=500,  # 指定Canvas组件的宽度
                    height=600,  # 指定Canvas组件的高度
                    bg='white')  # 指定Canvas组件的背景色
# im = Tkinter.PhotoImage(file='img.gif')     # 使用PhotoImage打开图片
image = IMG.open("bg.jpg")
im = ImageTk.PhotoImage(image)

canvas.create_image(300, 50, image=im)  # 使用create_image将图片添加到Canvas组件中
# canvas.create_text(302, 77,  # 使用create_text方法在坐标（302，77）处绘制文字
#                    text='Use Canvas'  # 所绘制文字的内容
#                    , fill='gray')  # 所绘制文字的颜色为灰色
# canvas.create_text(300, 75,
#                    text='Use Canvas',
#                    fill='blue')
canvas.pack()  # 将Canvas添加到主窗口

#采集数据库数据
def get():
    global pic, x_train, x_test, y_train, y_test, x_train1, x_test1, y_train1, y_test1
    global x_train0, x_test0, y_train0, y_test0, target_n
    global img_size
    global pca,clf
    pic = get_eigenvalue()
    print(pic.get_pca_eigen())
    print("------------------------------------------")

    x_train, x_test, y_train, y_test = pic.get_data()  # 存放图像降维后的灰度特征向量
    x_train1, x_test1, y_train1, y_test1 = pic.get_colordata()  # 存放图像降维前的彩色特征向量
    x_train0, x_test0, y_train0, y_test0 = pic.get_graydata()  # 存放图像降维前的灰度特征向量
    img_size = pic.get_img_size()  # 记录各图片的h，w
    target_n = pic.get_target_n()  # 存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号
    pca = pic.get_pca()
    clf = SVM1("rbf", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], pic)



def plot_gallary():

    # pic = get_eigenvalue()
    # print(pic.get_pca_eigen())
    # print("------------------------------------------")
    # x_train, x_test, y_train, y_test = pic.get_data()  # 存放图像降维后的灰度特征向量
    # x_train1, x_test1, y_train1, y_test1 = pic.get_colordata()  # 存放图像降维前的彩色特征向量
    # x_train0, x_test0, y_train0, y_test0 = pic.get_graydata()  # 存放图像降维前的灰度特征向量
    # img_size = pic.get_img_size()  # 记录各图片的h，w
    # target_n = pic.get_target_n()  # 存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号
    if target_n == None:
        print ("请先采集数据！")
        return 0

    # 预测图片
    print("------------------------------------------")
    print("正在预测图片...")
    t0 = time()
    test_pred = clf.predict(x_test)
    print("完成预测，共计 %0.3f秒" % (time() - t0))
    print("------------------------------------------")
    print("预测结果如下：")
    print(classification_report(y_test.astype(str), test_pred.astype(str), target_names=target_n))
    n_classes = pic.get_nclasses()
    print(confusion_matrix(y_test, test_pred, labels=range(n_classes)))
    print("------------------------------------------")
    prediction = 0
    for i in range(len(test_pred)):
        if test_pred[i] == y_test[i]:
            prediction += 1
    precision_average = prediction / len(y_test)

    # test_pred存放预测结果，对应y_test
    print("预测的准确率为：", precision_average)
    result = []
    for i in range(len(test_pred)):
        result.append({"target:": target_n[y_test[i]], "predict:": target_n[test_pred[i]]})
    for i in range(100):
        print(result[i])

    # 比较预测值和实际值
    def plot_gallery(images, titles, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(2.5 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35, wspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            # plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            # image = images[i].reshape(h, w)
            # img = []
            # for j in range(len(image)):
            #     img.append(list(image[j]))
            # #print(img[:5])
            plt.imshow(images[i])
            # plt.imshow(images[i])
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    def title(y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        if y_pred[i] == y_test[i]:
            pre = "(√)"
        else:
            pre = '(×)'
        return 'predicted: %s%s\ntrue:      %s' % (pred_name, pre, true_name)

    def show_prediction():
        prediction_titles = [title(test_pred, y_test, target_n, i)
                             for i in range(len(test_pred))]

        plot_gallery(x_test1, prediction_titles)

        # plot the gallery of the most significative eigenfaces

        # eigenface_titles = ["eigenface %d" % i for i in range(len(eigenfaces))]
        # plot_gallery(eigenfaces, eigenface_titles, h = 30, w = 18)

        plt.show()

    show_prediction()


def show_precise():
    # pic = get_eigenvalue()
    # print (pic.get_pca_eigen())
    # print("------------------------------------------")
    # x_train, x_test, y_train, y_test = pic.get_data()#存放图像降维后的灰度特征向量
    # x_train1, x_test1, y_train1, y_test1 = pic.get_colordata()#存放图像降维前的彩色特征向量
    # x_train0, x_test0, y_train0, y_test0 = pic.get_graydata()#存放图像降维前的灰度特征向量
    # img_size = pic.get_img_size()#记录各图片的h，w
    # target_n = pic.get_target_n()#存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号

    # test_pred, precision_average = SVM1("rbf", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], pic)
    # #test_pred存放预测结果，对应y_test
    # print ("预测的准确率为：", precision_average)
    # result = []
    # for i in range(len(test_pred)):
    #     result.append((y_test[i], test_pred[i]))
    if target_n == None:
        print("请先采集数据！")
        return 0
    #绘制预测准确度的曲线
    def show_pecise():
        kernel_to_test = ['rbf', 'poly', 'sigmoid']

        for kernel_name in kernel_to_test:
            x_label = np.linspace(0.0001, 1, 100)
            y_label = []
            for i in x_label:
                a, b = SVM2(kernel_name, [i], pic)
                y_label.append(b)
            plt.plot(x_label, y_label, label=kernel_name)

        plt.xlabel("Gamma")
        plt.ylabel("Precision")
        plt.title('Different Kernels Contrust')
        plt.legend()
        plt.show()
    show_pecise()

def select():
    global filename
    filename = filedialog.askopenfilename(filetypes = [('JPG', '.jpg'), ('PNG', '.png'), ('BMP', '.bmp')])
    print (filename)


def predict():
    if filename == None:
        print('请先选择图片！')
        return 0
    if target_n == None:
        print ("请先采集数据！")
        return 0
    img0 = []
    im = IMG.open(filename)
    #self.pic_data_rgb.append(img.resize((50, 37)))
    #self.img_size.append(img.size)
    img = im.convert('L')  # 原来的data是[(r,g,b)],现转为灰度图像[grey]
    img = img.resize((400, 200))  # 将图片转化为统一的格式，统一之后操作数组的长度
    img = img.getdata()
    img0.append(list(img))
    img = pca.transform(img0)
    img = np.array(img)
    prediction = clf.predict(img)
    print ('I guess you chose : ' + target_n[prediction[0]])
    # tkimg = ImageTk.PhotoImage(im)  # 执行此函数之前， Tk() 必须已经实例化。
    # l = Label(Frame3, image=tkimg)
    # l.grid(row = 3, column = 0, sticky = NW, pady = 8, padx = 20)
    # l.show()
    plt.imshow(im)
    plt.title("I guess this is..." + target_n[prediction[0]] + "?" )
    plt.axis('off')  # clear x- and y-axes
    plt.show()
    #im.show()  # 显示刚才所画的所有操作



# b0 = Button(Frame1, text = "采集数据库", command = get, width=30, height=2, bd = 5)
# b1 = Button(Frame1, text = "显示预测结果", command = plot_gallary, width=30, height=2,bd = 5)
# b2 = Button(Frame1, text = "显示预测准确度", command = show_precise, width=30, height=2, bd = 5)
# b3 = Button(Frame1, text = "退出程序", command = root.quit, width=30, height=2, bd = 5)
# b4 = Button(Frame2, text = "选择图片", command = select, width=30, height=2, bd = 5)
# b5 = Button(Frame2, text = "预测", command = predict, width=30, height=2, bd = 5)
# b0.pack()
# b1.pack()
# b2.pack()
# b3.pack()
# b4.pack()
# b5.pack()

filemenue = Menu(menubar, tearoff = False)
filemenue.add_command(label = "采集数据", command = get)
filemenue.add_separator()
filemenue.add_command(label = "退出", command = root.quit)
menubar.add_cascade(label = "Start", menu = filemenue)

testmenue = Menu(menubar, tearoff = False)
testmenue.add_command(label = "预测结果", command = plot_gallary)
testmenue.add_command(label = "预测精度", command = show_precise)
menubar.add_cascade(label = "Test", menu = testmenue)

choosemenu = Menu(menubar, tearoff = False)
choosemenu.add_command(label = "选择图片", command = select)
choosemenu.add_command(label = "预测", command = predict)
menubar.add_cascade(label = "FreeRun", menu = choosemenu)


root.config(menu = menubar)

mainloop()