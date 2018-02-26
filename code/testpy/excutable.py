#!/usr/bin/python

from tkinter import Tk, RIGHT, BOTH, RAISED, Frame, Button, Label, font, Text,Scrollbar,LEFT, HORIZONTAL, Entry, END
from tkinter.ttk import  Style
import classifier

# Code to add widgets will go here...
class Example(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")

        self.parent = parent

        self.initUI()


    def initUI(self):

        self.parent.title("Sentiment Analysis_Chen, Lan, Zhang, Zheng")
       ## self.pack(fill=BOTH, expand=1)
        self.style = Style()
        self.style.theme_use("default")
        self.centerWindow()


        self.pack(fill=BOTH, expand=True)
        lbl = Label(self, text="Input Text",width=20, height=2,background="#FFFFFF")
        lbl.config(font=("Elephant", 12))
        lbl.place(x = 140, y=5)

        txt_frm = Frame(self, width=400, height=200)
        txt_frm.place(x = 50, y = 50)
        # ensure a consistent GUI size
        txt_frm.grid_propagate(False)
        # implement stretchability
        txt_frm.grid_rowconfigure(0, weight=1)
        txt_frm.grid_columnconfigure(0, weight=1)

    # create a Text widget
        self.Intxt = Text(txt_frm, borderwidth=3, relief="sunken")
        self.Intxt.config(font=("consolas", 12), undo=True, wrap='word')
        self.Intxt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        scrollby = Scrollbar(txt_frm, command=self.Intxt.yview)
        scrollby.grid(row=0, column=1, sticky='nsew')

        self.Intxt['yscrollcommand'] = scrollby.set

        txt_frm1 = Frame(self, width=400, height=100)
        txt_frm1.place(x = 50, y = 250)
        # ensure a consistent GUI size
        txt_frm1.grid_propagate(False)
        # implement stretchability
        txt_frm1.grid_rowconfigure(0, weight=1)
        txt_frm1.grid_columnconfigure(0, weight=1)

    # create a Text widget
        self.outtxt = Text(txt_frm1,borderwidth=3, relief="sunken")
        self.outtxt.config(font=("consolas", 12), undo=True, wrap='word')
        self.outtxt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    # create a Scrollbar and associate it with txt
        scrollby1 = Scrollbar(txt_frm1, command=self.outtxt.yview)
        scrollby1.grid(row=0, column=1, sticky='nsew')

        self.outtxt['yscrollcommand'] = scrollby1.set
        quitButton = Button(self,height=1,width=20, text="Quit", command=self.quit)
        quitButton.place(x=180, y=460)
        ##quitButton.pack(side=RIGHT, padx=20, pady=5)
        okButton = Button(self, height=1,width=20, text="Run", command=self.runscript)
        okButton.place(x=180, y = 400)

    def centerWindow(self):
        w = 500
        h = 500

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2 -100
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))


    def runscript(self):
        inputt = self.Intxt.get(1.0,END)
        file = open("in.txt", "w")
        file.write(inputt)
        file.close()
        #print(inputt)
        split = inputt.splitlines();
        #test = ''.join(inputt)
        # counter = 0;
        # test = {};
        # for line in split:
        #     if line != "":
        #         test[counter] = line
        #         counter = counter + 1;
        # test = "\n".join(test)
        res = classifier.run()
        self.outtxt.delete(1.0, END)
        for result in res:
            result = result + 1;
            self.outtxt.insert(END, result)
            self.outtxt.insert(END, "\n")       

def main():
    root = Tk()
    root.geometry("500x500+300+300")
    app = Example(root)
    font.families()
    root.mainloop()

if __name__ == '__main__':
    main()
