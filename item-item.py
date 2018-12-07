# -*- coding: utf-8 -*-
from tkinter import *
from tkinter.ttk import *
from tkinter import scrolledtext
from pyspark import SparkContext
from collections import defaultdict
import numpy as np
from itertools import combinations

# Creating Tinker object
window = Tk()
window.title("Welcome to Beer Recommendation System")
window.geometry('360x360')
lbl = Label(window, text="FILE1")
lbl.grid(column=0, row=1)
txt = Entry(window, width=20)
txt.grid(column=1, row=1)
lbl01 = Label(window, text="FILE2")
lbl01.grid(column=0, row=2)
txt01 = Entry(window, width=20)
txt01.grid(column=1, row=2)
lbl02 = Label(window, text="USERNAME")
lbl02.grid(column=0, row=3)
txt02 = Entry(window, width=20)
txt02.grid(column=1, row=3)
lbl1 = Label(window, text="AROMA")
lbl1.grid(column=0, row=4)
txt2 = Entry(window, width=10)
txt2.grid(column=1, row=4)
lbl2 = Label(window, text="APPEARANCE")
lbl2.grid(column=0, row=5)
txt3 = Entry(window, width=10)
txt3.grid(column=1, row=5)
lbl3 = Label(window, text="PALLATE")
lbl3.grid(column=0, row=6)
txt4 = Entry(window, width=10)
txt4.grid(column=1, row=6)
lbl4 = Label(window, text="TASTE")
lbl4.grid(column=0, row=7)
txt5 = Entry(window, width=10)
txt5.grid(column=1, row=7)


def clicked():
    global res0
    global res01
    global res02
    global res1
    global res2
    global res3
    global res4
    res0 = txt.get()
    res01 = txt01.get()
    res02 = txt02.get()
    res1 = txt2.get()
    res2 = txt3.get()
    res3 = txt4.get()
    res4 = txt5.get()


btn = Button(window, text="ENTER", command=clicked)
btn.grid(column=0, row=12)
# Quit button
btn1 = Button(window, text="QUIT ONCE YOU PRESS ENTER", command=window.destroy)
btn1.grid(column=0, row=13)
window.mainloop()

sc = SparkContext()
lines = sc.textFile(res0)
# train,test = lines.randomSplit(weights=[1.0, 0.00], seed=1)

# weight-aroma =sys.argv[3]
# weight-appearance =sys.argv[4]
# weight-palate =sys.argv[5]
# weight-taste =sys.argv[6]
aroma = res1
appearance = res2
palate = res3
taste = res4


def getdata(line):
    line = line.split(",")
    total_rating_num = (
                (2 * float(line[0])) + (float(aroma) * float(line[1])) + (float(appearance) * float(line[2])) + (
                    float(palate) * float(line[3])) + (float(taste) * float(line[4])))
    total_rating_denom = (2 + int(aroma) + int(appearance) + int(palate) + int(taste))
    total_rating = float(total_rating_num) / float(total_rating_denom)
    print(total_rating)
    return (line[5]), ((line[6]), float(total_rating))


user_beer_ratings = lines.map(lambda x: getdata(x))
user_beer_tuples = user_beer_ratings.groupByKey()


def beerpairs(userid, beersrated):
    rated = beersrated
    """
    for i in rated:
        j=i+1
        if j:
            return (i[0],[0]),(i[1],j[1])
            """

    for i, j in combinations(rated, 2):
        return (i[0], j[0]), (i[1], j[1])


create_beerpairs = user_beer_tuples.filter(lambda x: len(x[1]) > 1).map(lambda x: beerpairs(x[0], x[1])).mapValues(
    lambda x: [x]).reduceByKey(lambda x, y: y + x)
beerpair_tuples = create_beerpairs.map(lambda x: (x[0], tuple(x[1])))


# print(beerpair_tuples)
def calculatecosine(pair, rating):
    total1 = 0
    total2 = 0
    dotproduct = 0
    x = 0
    for eachrating in rating:
        total1 += float((eachrating[0]) * (eachrating[0]))
    for eachrating in rating:
        total2 += float((eachrating[1]) * (eachrating[1]))
    for eachrating in rating:
        dotproduct += float((eachrating[0]) * (eachrating[1]))
        x += 1
    cosinesim_denominator = (np.sqrt(total1) * np.sqrt(total2))
    cosinesim_numerator = dotproduct
    if cosinesim_denominator:
        cosinesim = float((cosinesim_numerator) / (cosinesim_denominator))
        return pair, (cosinesim)
    else:
        return pair, (0.0)


beerswithcosineval = beerpair_tuples.map(lambda x: calculatecosine(x[0], x[1]))


# print(beerswithcosineval)
def makeeachgroup(beer_pair, cosinevalues):
    (beer1, beer2) = beer_pair
    return beer1, (beer2, cosinevalues)


beersandtheirgroups = beerswithcosineval.map(lambda x: makeeachgroup(x[0], x[1])).mapValues(lambda x: [x]).reduceByKey(
    lambda x, y: x + y)
beersandtheirgroups_tuples = beersandtheirgroups.map(lambda x: (x[0], tuple(x[1])))


def gettopsimilar(beer1, similaritygroup):
    similaritygroup.sort(key=lambda x: x[1], reverse=True)
    return beer1, similaritygroup


beersimilaritems = beersandtheirgroups_tuples.map(lambda x: (x[0], list(x[1])))
beersimilaritems = beersimilaritems.map(lambda x: gettopsimilar(x[0], x[1])).collect()

beerdictionary = {}
for (beer, similarbeers) in beersimilaritems:
    beerdictionary[beer] = similarbeers

i = sc.broadcast(beerdictionary)


def recommend(user, beerwithrating, beerdictionary, n):
    pred_numerator = defaultdict(int)
    pred_denominator = defaultdict(int)
    for (beer, rating) in beerwithrating:
        similarbeers = beerdictionary.get(beer, None)
        if similarbeers:
            for (similarbeer, (cosinevalue)) in similarbeers:
                if similarbeer != beer:
                    pred_numerator[similarbeer] += cosinevalue * rating
                    pred_denominator[similarbeer] += cosinevalue

    # create the normalized list of scored movies
    predictedratings = [(float(total / pred_denominator[beer]), beer) for beer, total in pred_numerator.items()]

    predictedratings.sort(reverse=True)

    return user, predictedratings[:n]


beer_recommend = user_beer_tuples.map(lambda p: recommend(p[0], p[1], i.value, 20)).reduceByKey(
    lambda x, y: x + y).collect()


# print(beer_recommend)
def mappingbeername(x):
    beernames = {}
    with open(x) as f:
        for line in f:
            x = line.split(',')
            beernames[int(x[0])] = x[1]
    return beernames


beernames = mappingbeername(res01)

uu = res02

userandrecommendations = {}

for (user, suggestions) in beer_recommend:
    userandrecommendations[user] = suggestions

beerlist = list()
print("Suggested Beers for the User - "+uu)
f = open("output-item.txt", "w")
for rating, suggestedbeer in userandrecommendations[uu]:
    suggestedbeer = beernames[int(suggestedbeer)]
    a = suggestedbeer,rating
    print a
    print >> f, a


f.close()
window1 = Tk()
window1.title("OUTPUT FOR BEER RECOMMENDATION")
window1.geometry('360x480')
lbl_final = Label(window1, text="")
lbl_final.grid(column=0, row=1)
final = "RESULTS FOR USERNAME - " + uu
lbl_final.configure(text=final)
# Scroll text window to populate output
txt_s = scrolledtext.ScrolledText(window1, width=50, height=30)
txt_s.grid(column=0, row=2)
with open("output-item.txt", 'r') as f:
    txt_s.insert(INSERT, f.read())
# Quit button
btn1 = Button(window1, text="QUIT", command=window1.destroy)
btn1.grid(column=0, row=4)
window1.mainloop()
sc.stop()