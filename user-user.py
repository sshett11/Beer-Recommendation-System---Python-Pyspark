from tkinter import *
from tkinter.ttk import *
from tkinter import scrolledtext
import numpy as np
import pandas as pd

# reading data from csv
df = pd.read_csv("beer_data_user-user.csv", sep=',', encoding='latin-1')

columns_review = ['beerid', 'beer_name', 'review_profilename', 'review_overall', 'review_aroma', 'review_palate',
                  'review_taste']
# print df.columns
df[columns_review].head()

# Creating Tinker object
window = Tk()
window.title("Welcome to Beer Recommendation System")
window.geometry('360x360')
lbl = Label(window, text="USERNAME")
lbl.grid(column=0, row=2)
txt = Entry(window, width=20)
txt.grid(column=1, row=2)
lbl1 = Label(window, text="AROMA")
lbl1.grid(column=0, row=3)
txt2 = Entry(window, width=10)
txt2.grid(column=1, row=3)
lbl2 = Label(window, text="PALLATE")
lbl2.grid(column=0, row=4)
txt3 = Entry(window, width=10)
txt3.grid(column=1, row=4)
lbl3 = Label(window, text="TASTE")
lbl3.grid(column=0, row=5)
txt4 = Entry(window, width=10)
txt4.grid(column=1, row=5)
lbl4 = Label(window, text="APPEARANCE")
lbl4.grid(column=0, row=6)
txt5 = Entry(window, width=10)
txt5.grid(column=1, row=6)


def clicked():
    global res
    global res1
    global res2
    global res3
    global res4
    res = txt.get()
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

weights_input = [int(res1), int(res2), int(res3), int(res4)]
# In[4]:
# function to calculate weighted rating


def weighted(weights):
    def weighted_rating(weights, overall, aroma, palate, taste, appearence):
        aroma = weights[0] * aroma
        palate = palate * weights[1]
        taste = weights[2] * taste
        appearence = weights[3] * appearence
        sum = 0.0
        for i in weights:
            sum = sum + i
        ratsum = (aroma + palate + taste)
        sum = sum * 2
        return overall / 2 + (ratsum) / sum

    df['weighted_rating'] = df.apply(
        lambda row: weighted_rating(weights, row['review_overall'], row['review_aroma'], row['review_palate'],
                                    row['review_taste'], row['review_appearance']), axis=1)


weighted(weights_input)

# In[5]:
# assigning unique beer ids and user ids

# df[['beerId','name','user_profileName','weighted_rating']]
df = df.assign(userid=(df.review_profilename).astype('category').cat.codes)
df = df.assign(beerid=(df.beerid).astype('category').cat.codes)
user_number = df.userid.unique().shape[0]
beer_number = df.beerid.unique().shape[0]

# building a rating matrix
data_matrix = np.zeros((user_number, beer_number))
for line in df.itertuples():
    data_matrix[line[10] - 1, line[11] - 1] = line[12]

#
num_users, num_items = data_matrix.shape
b = np.mean(data_matrix[np.where(data_matrix != 0)])


class MatixFactorization():

    def __init__(self, Rating, la_dim, al, beta, no_of_interations):
        # al is learning rate , beta is regularization, la_dim is number of latent
        # dimensions in the p and Q matrix
        self.al = al
        self.beta = beta
        self.no_of_iterations = no_of_interations
        self.Rating = Rating
        self.no_of_users, self.no_of_items = Rating.shape
        self.la_dim = la_dim

    # whole model flow
    def model(self):
        self.user_bias = np.zeros(self.no_of_users)  # biases initialization
        self.item_bias = np.zeros(self.no_of_items)
        self.mean_bias = np.mean(self.Rating[np.where(self.Rating != 0)])
        self.samples = [(i, j, self.Rating[i, j]) for i in range(self.no_of_users) for j in range(self.no_of_items) if
                        self.Rating[i, j] > 0]
        training_output = []
        self.P = np.random.normal(scale=1. / self.la_dim, size=(self.no_of_users, self.la_dim))
        self.Q = np.random.normal(scale=1. / self.la_dim, size=(self.no_of_items, self.la_dim))
        # creating initial P and Q matrices

        for i in range(self.no_of_iterations):  # running stochastic gradient for
            # number of iterations
            np.random.shuffle(self.samples)
            self.sto_grad()
            mean_sq_error = self.mean_err()
            training_output.append((i, mean_sq_error))
            if (i + 1) % 5 == 0:
                print("Iteration Number: %d ; error = %.4f" % (i + 1, mean_sq_error))

        return training_output

    def sto_grad(self):  # stochastic gradient
        for i, j, r in self.samples:
            prediction = self.cal_rating(i, j)
            err = (r - prediction)
            # bias update
            self.user_bias[i] += self.al * (err - self.beta * self.user_bias[i])
            self.item_bias[j] += self.al * (err - self.beta * self.item_bias[j])
            # latentfeature update
            self.P[i, :] += self.al * (err * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.al * (err * self.P[i, :] - self.beta * self.Q[j, :])

    def cal_rating(self, i, j):
        prediction = self.mean_bias + self.user_bias[i] + self.item_bias[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def mean_err(self):
        # mean square error calculation
        prediction = self.matrix_complete()
        error = 0
        xs, ys = self.Rating.nonzero()
        for x, y in zip(xs, ys):
            error += pow(self.Rating[x, y] - prediction[x, y], 2)
        return np.sqrt(error)

    def matrix_complete(self):  # returns P*Q
        return self.mean_bias + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis:, ] + self.P.dot(self.Q.T)


# In[32]:


mf = MatixFactorization(data_matrix, 6, 0.02, 0.01, 150)

# In[33]:


mf.model()
recom_mat = mf.matrix_complete()


# In[25]:
def max_item_index(a):
    c = []
    for i in range(10):
        maxind = a.index(max(a))
        c.append(maxind)
        a[maxind] = 0
    return c


def get_user_recommedation(user_name):
    usrid = df[df["review_profilename"] == user_name].userid.head().tolist()[0]
    userlist_np = recom_mat[usrid]
    userlist = userlist_np.tolist()
    beerid_list = max_item_index(userlist)
    recommend_list_beer = []
    for brid in beerid_list:
        beername = df[df["beerid"] == brid].beer_name.head().tolist()[0]
        recommend_list_beer.append(beername)
    return recommend_list_beer


# In[ ]:
user_name = res
output = get_user_recommedation(user_name)
f = open("output-user.txt", "w")
print "here are the recommendation for the user:"
print " "
for a in output:
    print a
    print >> f, a
    print " "

f.close()
window1 = Tk()
window1.title("OUTPUT FOR BEER RECOMMENDATION")
window1.geometry('360x480')
lbl_final = Label(window1, text="")
lbl_final.grid(column=0, row=1)
final = "RESULTS FOR USERNAME - " + user_name
lbl_final.configure(text=final)
# Scroll text window to populate output
txt_s = scrolledtext.ScrolledText(window1, width=50, height=30)
txt_s.grid(column=0, row=2)
with open("output-user.txt", 'r') as f:
    txt_s.insert(INSERT, f.read())
# Quit button
btn1 = Button(window1, text="QUIT", command=window1.destroy)
btn1.grid(column=0, row=3)
window1.mainloop()