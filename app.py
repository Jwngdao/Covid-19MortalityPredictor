try:
    import tkinter as tk
except:
    import tkinter as tk

from tkinter import *
from PIL import Image,ImageTk
from tkinter import messagebox
from tkinter.filedialog import askopenfile

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("COVID_10000.csv")
data=data.drop_duplicates(keep='first')
health = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
doa = data.iloc[:, 18]

a = []

class Window(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(startPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class startPage(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        tk.Label(self, text="Welcome To COVID'19 Mortality prediction with Machine Learning Techniques",
                 font=('Helvetica', 18, "bold")).pack(side="top", fill="x", pady=5)
        tk.Button(self, text="USER INPUT", command=lambda: root.switch_frame(PageOne)).pack()
        tk.Button(self, text="CSV INPUT", command=lambda: root.switch_frame(PageTwo)).pack()

class PageOne(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)

        entry1 = open('COVID_10000.csv', 'a')

        Label(root, text="Gender").pack()
        a = tk.Entry(root)
        a.pack()

        Label(root, text="Intubed").pack()
        b = tk.Entry(root)
        b.pack()

        Label(root, text="Pneumonia").pack()
        c = tk.Entry(root)
        c.pack()

        Label(root, text="Age").pack()
        d = tk.Entry(root)
        d.pack()

        Label(root, text="Pregnancy").pack()
        e = tk.Entry(root)
        e.pack()

        Label(root, text="Diabetes").pack()
        f = Entry(root)
        f.pack()

        Label(root, text="COPD").pack()
        g = tk.Entry(root)
        g.pack()

        Label(root, text="Asthma").pack()
        h = tk.Entry(root)
        h.pack()

        Label(root, text="Inmsupr").pack()
        i = tk.Entry(root)
        i.pack()

        Label(root, text="Hypertension").pack()
        j = tk.Entry(root)
        j.pack()

        Label(root, text="Other Diseases").pack()
        k = tk.Entry(root)
        k.pack()

        Label(root, text="Cardiovascular").pack()
        l = tk.Entry(root)
        l.pack()

        Label(root, text="Obesity").pack()
        m = tk.Entry(root)
        m.pack()

        Label(root, text="Renal Chronic").pack()
        n = tk.Entry(root)
        n.pack()

        Label(root, text="Tobacco").pack()
        o = tk.Entry(root)
        o.pack()

        Label(root, text="Contact with other Covid patient").pack()
        p = tk.Entry(root)
        p.pack()

        Label(root, text="Covid Result").pack()
        q = tk.Entry(root)
        q.pack()

        Label(root, text="ICU").pack()
        r = tk.Entry(root)
        r.pack()

        def entry():

            test = []
            test.append(a.get())
            test.append(b.get())
            test.append(c.get())
            test.append(d.get())
            test.append(e.get())
            test.append(f.get())
            test.append(g.get())
            test.append(h.get())
            test.append(i.get())
            test.append(j.get())
            test.append(k.get())
            test.append(l.get())
            test.append(m.get())
            test.append(n.get())
            test.append(o.get())
            test.append(p.get())
            test.append(q.get())
            test.append(r.get())

            entry1.write(a.get())
            entry1.write(",")

            entry1.write(b.get())
            entry1.write(",")

            entry1.write(c.get())
            entry1.write(",")

            entry1.write(d.get())
            entry1.write(",")

            entry1.write(e.get())
            entry1.write(",")

            entry1.write(f.get())
            entry1.write(",")

            entry1.write(g.get())
            entry1.write(",")

            entry1.write(h.get())
            entry1.write(",")

            entry1.write(i.get())
            entry1.write(",")

            entry1.write(j.get())
            entry1.write(",")

            entry1.write(k.get())
            entry1.write(",")

            entry1.write(l.get())
            entry1.write(",")

            entry1.write(m.get())
            entry1.write(",")

            entry1.write(n.get())
            entry1.write(",")

            entry1.write(o.get())
            entry1.write(",")

            entry1.write(p.get())
            entry1.write(",")

            entry1.write(q.get())
            entry1.write(",")

            entry1.write(r.get())
            entry1.write(",")

            test1 = []
            test1.append(test)

            # enter in csv

            outputs = []

            def dtc():
                dt = DecisionTreeClassifier()
                dt.fit(health, doa)
                dt_pred = dt.predict(test1)
                if dt_pred == 0:
                    outputs.append(0)
                else:
                    outputs.append(1)
                return outputs

            def logr():
                scX = StandardScaler()
                health1 = scX.fit_transform(health)
                lr = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
                lr.fit(health1, doa.values.ravel())
                test2 = scX.transform(test1)
                lr_pred = lr.predict(test2)
                if lr_pred == 0:
                    outputs.append(0)
                else:
                    outputs.append(1)
                return outputs

            def knn():
                sc = StandardScaler()
                health1 = sc.fit_transform(health)
                test2 = sc.transform(test1)
                knearest = KNeighborsClassifier()
                knearest.fit(health1, doa)
                knn_pred = knearest.predict(test2)
                if knn_pred == 0:
                    outputs.append(0)
                else:
                    outputs.append(1)
                return outputs

            def rfc():
                rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
                rf.fit(health, np.ravel(doa))
                rf_pred = rf.predict(test1)
                if rf_pred == 0:
                    outputs.append(0)
                else:
                    outputs.append(1)
                return outputs

            def nb():
                sc = StandardScaler()
                health1 = sc.fit_transform(health)
                test2 = sc.transform(test1)
                nb = GaussianNB()
                nb.fit(health1, doa.values.ravel())
                nb_pred = nb.predict(test2)
                if nb_pred == 0:
                    outputs.append(0)
                else:
                    outputs.append(1)
                return outputs

            def res():
                dtc()
                logr()
                knn()
                rfc()
                nb()
                final_output = max(outputs, key=outputs.count)

                if final_output == 0:
                    entry1.write('0')
                    messagebox.showinfo("Patient's Result", "Health Condition of Patient is Mild.")

                else:
                    entry1.write('1')
                    messagebox.showinfo("Patient's Result", "Health Condition of Patient is Serious.")

                entry1.write("\n")

            if button:
                res()

        button = Button(self, text="SUBMIT", width=10, command=entry)
        button.pack()
        tk.Button(self, text="Back", command=lambda: root.switch_frame(startPage)).pack()


class PageTwo(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)

        def open_file():
            browse_csv.set('LOADING.....')
            file = askopenfile(parent=root, mode='rb', title='Choose a CSV File', filetype=[("CSV File", "*.csv")])
            if file:
                new_data = pd.read_csv(file)
                new_data1 = new_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
                entry = open('COVID_10000.csv', 'a')
                d = []
                l = []
                k = []
                r = []
                n = []

                def dtc():
                    dt = DecisionTreeClassifier()
                    dt.fit(health, doa)
                    dt_pred = dt.predict(new_data)
                    for i in dt_pred:
                        d.append(i)
                    return d

                def logisitic():
                    scX = StandardScaler()
                    health1 = scX.fit_transform(health)
                    lr = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
                    lr.fit(health1, doa.values.ravel())
                    test2 = scX.transform(new_data)
                    lr_pred = lr.predict(test2)
                    for i in lr_pred:
                        l.append(i)
                    return l

                def knn():
                    sc = StandardScaler()
                    health1 = sc.fit_transform(health)
                    test2 = sc.transform(new_data)
                    knearest = KNeighborsClassifier()
                    knearest.fit(health1, doa)
                    knn_pred = knearest.predict(test2)
                    for i in knn_pred:
                        k.append(i)
                    return k

                def rfc():
                    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
                    rf.fit(health, np.ravel(doa))
                    rf_pred = rf.predict(new_data)
                    for i in rf_pred:
                        r.append(i)

                    return r

                def nb():
                    sc = StandardScaler()
                    health1 = sc.fit_transform(health)
                    test2 = sc.transform(new_data)
                    nb = GaussianNB()
                    nb.fit(health1, doa.values.ravel())
                    nb_pred = nb.predict(test2)
                    for i in nb_pred:
                        n.append(i)
                    return n

                def arr_com():
                    dtc()
                    logisitic()
                    knn()
                    rfc()
                    nb()
                    a = list(zip(d, l, k, r, n))
                    b = list(map(list, a))
                    y = 1
                    results = ''
                    for i, j in zip(b, new_data1):
                        f_out = max(i, key=i.count)
                        for x in j:
                            entry.write(str(x))
                            entry.write(',')
                        entry.write(str(f_out))
                        entry.write('\n')

                        if f_out == 0:
                            # textbox
                            # text_box=tk.Text(root,height=100,width=30, padx=15,pady=15)
                            # text_box.insert(1.0,f'Health Condition of Patient No. {y} is Mild".\n' )
                            # text_box.grid(column=1,row=3)
                            text = f'Health Condition of Patient No. {y} is Mild.\n'
                            results += text


                        else:
                            # textbox
                            # text_box=tk.Text(root,height=100,width=30, padx=15,pady=15)
                            # text_box.insert(1.0,f'Health Condition of Patient No. {y} is Serious".\n' )
                            # text_box.grid(column=1,row=3)
                            text = f'Health Condition of Patient No. {y} is Serious.\n'
                            results += text

                        y += 1

                    # textbox
                    text_box = tk.Text(root, height=600, width=100, padx=30, pady=30)
                    text_box.insert(1.0, results)
                    text_box.grid(column=1, row=0)
                    # return b

                arr_com()

        browse_csv = tk.StringVar()
        browse_btn = tk.Button(root, textvariable=browse_csv, command=lambda: open_file(), font='Raleway', bg="#20bebe",
                               fg='white', height=2, width=15)
        browse_csv.set('BROWSE')
        browse_btn.pack()


root = tk.Tk()


app = Window()
root.mainloop()



