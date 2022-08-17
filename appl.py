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

root=tk.Tk()
root.title("covid")
root.geometry("600x1100")
#logo
def logo():
    canvas=tk.Canvas(root,width=600, height=300)
    canvas.grid(columnspan=3)
    logo=Image.open('logo.ico')
    logo=ImageTk.PhotoImage(logo)
    logo_label=tk.Label(image=logo)
    logo_label.image=logo
    logo_label.place(relx=.5,rely=.18,anchor=CENTER)
logo()

author=Label(root, text="Built by: Binita, Jwngdao & Bibung",bg="#ffff00",fg='black')
author.place(relx=.51,rely=.97,anchor=CENTER)

data = pd.read_csv("COVID_10000.csv")
health = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
doa = data.iloc[:, 18]
a = []
def user_input():
    entry1 = open('COVID_10000.csv', 'a')
    def clear_label():
        a1.destroy()
        b1.destroy()
        c1.destroy()
        d1.destroy()
        e1.destroy()
        f1.destroy()
        g1.destroy()
        h1.destroy()
        i1.destroy()
        j1.destroy()
        k1.destroy()
        l1.destroy()
        m1.destroy()
        n1.destroy()
        o1.destroy()
        p1.destroy()
        q1.destroy()
        r1.destroy()
    def clear_boxes():
        a.destroy()
        b.destroy()
        c.destroy()
        d.destroy()
        e.destroy()
        f.destroy()
        g.destroy()
        h.destroy()
        i.destroy()
        j.destroy()
        k.destroy()
        l.destroy()
        m.destroy()
        n.destroy()
        o.destroy()
        p.destroy()
        q.destroy()
        r.destroy()
    a1=Label(root, text="Gender")
    a1.grid(row=1, sticky=W)
    a = tk.Entry(root)
    a.grid(row=1, column=1)
    b1=Label(root, text="Intubed")
    b1.grid(row=2, sticky=W)
    b = tk.Entry(root)
    b.grid(row=2, column=1)
    c1=Label(root, text="Pneumonia")
    c1.grid(row=3, sticky=W)
    c = tk.Entry(root)
    c.grid(row=3, column=1)
    d1=Label(root, text="Age")
    d1.grid(row=4, sticky=W)
    d = tk.Entry(root)
    d.grid(row=4, column=1)
    e1=Label(root, text="Pregnancy")
    e1.grid(row=5, sticky=W)
    e = tk.Entry(root)
    e.grid(row=5, column=1)
    f1=Label(root, text="Diabetes")
    f1.grid(row=6, sticky=W)
    f = Entry(root)
    f.grid(row=6, column=1)
    g1=Label(root, text="COPD")
    g1.grid(row=7, sticky=W)
    g = tk.Entry(root)
    g.grid(row=7, column=1)
    h1=Label(root, text="Asthma")
    h1.grid(row=8, sticky=W)
    h = tk.Entry(root)
    h.grid(row=8, column=1)
    i1=Label(root, text="Inmsupr")
    i1.grid(row=9, sticky=W)
    i = tk.Entry(root)
    i.grid(row=9, column=1)
    j1=Label(root, text="Hypertension")
    j1.grid(row=10, sticky=W)
    j = tk.Entry(root)
    j.grid(row=10, column=1)
    k1=Label(root, text="Other Diseases")
    k1.grid(row=11, sticky=W)
    k = tk.Entry(root)
    k.grid(row=11, column=1)
    l1=Label(root, text="Cardiovascular")
    l1.grid(row=12, sticky=W)
    l = tk.Entry(root)
    l.grid(row=12, column=1)
    m1=Label(root, text="Obesity")
    m1.grid(row=13, sticky=W)
    m =tk.Entry(root)
    m.grid(row=13, column=1)
    n1=Label(root, text="Renal Chronic")
    n1.grid(row=14, sticky=W)
    n = tk.Entry(root)
    n.grid(row=14, column=1)
    o1=Label(root, text="Tobacco")
    o1.grid(row=15, sticky=W)
    o = tk.Entry(root)
    o.grid(row=15, column=1)
    p1=Label(root, text="Contact with other Covid patient")
    p1.grid(row=16, sticky=W)
    p = tk.Entry(root)
    p.grid(row=16, column=1)
    q1=Label(root, text="Covid Result")
    q1.grid(row=17, sticky=W)
    q = tk.Entry(root)
    q.grid(row=17, column=1)
    r1=Label(root, text="ICU")
    r1.grid(row=18, sticky=W)
    r = tk.Entry(root)
    r.grid(row=18, column=1)
    def go_to_next_entry(event, entry_list, this_index):
        next_index = (this_index + 1) % len(entry_list)
        entry_list[next_index].focus_set()
    entries = [child for child in root.winfo_children() if isinstance(child, Entry)]
    for idx, entry in enumerate(entries):
        entry.bind('<Return>', lambda e, idx=idx: go_to_next_entry(e, entries, idx))


    def write():
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
        test1 = []
        test1.append(test)
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

                write()
                entry1.write('0')
                entry1.write("\n")

                clear_boxes()
                clear_label()
                submit_btn.destroy()
                #text="Health Condition of Patient is Mild."
                r0=Label(root, text="Health Condition of Patient is Mild.")
                r0.grid(row=18,column=1)

                # exit_btn=tk.Button(root,text="Exit",command=lambda:root.destroy(),font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
                # exit_btn.place(relx=.93,rely=.37,anchor=CENTER)

                #home_button = tk.Button(root, text="HOME", command=lambda:[home(),home_button.destroy(),back_button.destroy(),r0.destroy(),clear_label(),clear_boxes()],font='Raleway',bg="#20bebe",fg='white',height=1,width=8)
                #home_button.place(relx=.82,rely=.37,anchor=CENTER)

                back_button = tk.Button(root, text="BACK", command=lambda:[user_input(),r0.destroy(),back_button.destroy(),clear_label(),clear_boxes(),home_button.destroy()],font='Raleway',bg="#000000",fg='white',height=1,width=8)
                back_button.place(relx=.07,rely=.37,anchor=CENTER)


            else:
                write()
                entry1.write('1')
                entry1.write("\n")

                clear_boxes()
                clear_label()
                submit_btn.destroy()
                #text="Health Condition of Patient is Serious."
                r1=Label(root, text="Health Condition of Patient is Serious.")
                r1.grid(row=18,column=1)


                #home_button = tk.Button(root, text="HOME", command=lambda:[home(),home_button.destroy(),r1.destroy(),back_button.destroy(),clear_label(),clear_label()],font='Raleway',bg="#20bebe",fg='white',height=1,width=8)
                #home_button.place(relx=.82,rely=.37,anchor=CENTER)

                back_button = tk.Button(root, text="BACK", command=lambda:[user_input(),r1.destroy(),back_button.destroy(),home_button.destroy(),clear_label(),clear_boxes()],font='Raleway',bg="#000000",fg='white',height=1,width=8)
                back_button.place(relx=.07,rely=.37,anchor=CENTER)

                # exit_btn=tk.Button(root,text="Exit",command=lambda:[root.destroy()],font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
                # exit_btn.place(relx=.93,rely=.37,anchor=CENTER)

        try:
            res()
        except:
            messagebox.showerror("Error!!","Invalid Inputs..!!! Please try again.")
            clear_boxes()
            clear_label()
            submit_btn.destroy()
            home_button.destroy()
            user_input()

    submit_btn=Button(root,text="SUBMIT",command=lambda:[entry(),back_button.destroy()],font='Raleway',bg="#00ff00",fg='white',height=1,width=8)
    submit_btn.place(relx=.87,rely=.93,anchor=CENTER)

    home_button = tk.Button(root, text="HOME", command=lambda:[home(),home_button.destroy(),submit_btn.destroy(),back_button.destroy(),clear_label(),clear_boxes()],font='Raleway',bg="#20bebe",fg='white',height=1,width=8)
    home_button.place(relx=.82,rely=.33,anchor=CENTER)

    back_button = tk.Button(root, text="BACK", command=lambda:[home(),back_button.destroy(),submit_btn.destroy(),clear_label(),clear_boxes(),home_button.destroy()],font='Raleway',bg="#000000",fg='white',height=1,width=8)
    back_button.place(relx=.07,rely=.33,anchor=CENTER)

    exit_btn=tk.Button(root,text="Exit",command=lambda:[root.destroy()],font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
    exit_btn.place(relx=.93,rely=.33,anchor=CENTER)



def csv_input():
    def open_file():
        browse_csv.set('LOADING.....')
        file=askopenfile(parent=root,mode='rb',title='Choose a CSV File',filetype=[("CSV File","*.csv")])
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
                y=1
                results=''
                for i, j in zip(b, new_data1):
                    f_out = max(i, key=i.count)
                    for x in j:
                        entry.write(str(x))
                        entry.write(',')
                    entry.write(str(f_out))
                    entry.write('\n')
                    if f_out==0:
                        text=f'Health Condition of Patient No. {y} is Mild.\n'
                        results+=text
                    else:
                        text=f'Health Condition of Patient No. {y} is Serious.\n'
                        results+=text
                    y+=1
                browse_btn.destroy()
                r1=Label(root, text=results)
                r1.place(relx=.3,rely=.36)


                back_button = tk.Button(root, text="BACK", command=lambda:[csv_input(),browse_btn.destroy(),r1.destroy(),back_button.destroy(),home_button.destroy()],font='Raleway',bg="#000000",fg='white',height=1,width=8)
                back_button.place(relx=.07,rely=.37,anchor=CENTER)
                #
                # exit_btn=tk.Button(root,text="Exit",command=lambda:[root.destroy()],font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
                # exit_btn.place(relx=.93,rely=.37,anchor=CENTER)
                #textbox
                #text_box=tk.Text(root,height=600,width=100, padx=30,pady=30)
                #text_box.insert(1.0,results )
                #text_box.grid(column=1,row=0)
            try:
                arr_com()
            except:
                messagebox.showerror("Error!!","Invalid Inputs...!! Please try again.")
                browse_btn.destroy()
                home_button.destroy()
                back_button.destroy()
                home_button.destroy()

                csv_input()


    browse_csv=tk.StringVar()
    browse_btn=tk.Button(root,textvariable=browse_csv,command=lambda:[open_file(),back_button.destroy()],font='Raleway',bg="#00ff00",fg='white',height=2,width=15)
    browse_csv.set('BROWSE')
    browse_btn.grid(column=1,row=1)

    home_button = tk.Button(root, text="HOME", command=lambda:[home(),browse_btn.destroy(),home_button.destroy(),exit_btn.destroy(),back_button.destroy()],font='Raleway',bg="#20bebe",fg='white',height=1,width=8)
    home_button.place(relx=.82,rely=.33,anchor=CENTER)

    back_button = tk.Button(root, text="BACK", command=lambda:[home(),browse_btn.destroy(),back_button.destroy(),home_button.destroy()],font='Raleway',bg="#000000",fg='white',height=1,width=8)
    back_button.place(relx=.07,rely=.33,anchor=CENTER)

    exit_btn=tk.Button(root,text="Exit",command=lambda:[root.destroy()],font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
    exit_btn.place(relx=.93,rely=.33,anchor=CENTER)
    #back_button = tk.Button(root, text="BACK", command=lambda :[home(),back_button.destroy(),browse_btn.destroy()],font='Raleway',bg="#ff0000",fg='white',height=1,width=12)
    #back_button.grid(column=1, row=2)
var=IntVar()
def home():
    R_btn1=tk.Button(root,text='USER INPUT',command=lambda:[user_input(),R_btn1.destroy(),R_btn2.destroy(),exit_btn.destroy()],font='Raleway',bg="#20bebe",fg='white',height=2,width=15)
    R_btn1.grid(column=1,row=2)
    R_btn2=tk.Button(root,text="CSV INPUT",command=lambda:[csv_input(),R_btn1.destroy(),R_btn2.destroy(),exit_btn.destroy()],font='Raleway',bg="#20bebe",fg='white',height=2,width=15)
    R_btn2.grid(column=1,row=3)
    exit_btn=tk.Button(root,text="Exit",command=lambda:root.destroy(),font='Raleway',bg="#FF0000",fg='white',height=1,width=8)
    exit_btn.place(relx=.93,rely=.33,anchor=CENTER)
home()
root.mainloop()