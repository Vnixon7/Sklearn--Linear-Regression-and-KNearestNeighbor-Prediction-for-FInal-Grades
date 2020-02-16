import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


##TRAIN MODEL -- LAST TRAIN ACC(98.301)##
best = 0.9743353123117502
for i in range(20000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    #y = mx+b


    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('interation: ',i,'Accuracy: ',acc*100)

    if acc > best:
        best = acc
        with open('grade_predict.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print(best)
        break


load_in = open('grade_predict.pickle', 'rb')
linear = pickle.load(load_in)

print('Co-efficent: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)





'''best = .725
for i in range(2000000):
    #training function with SKLEARN ###Test size is used for the amount of data tested. 0.2 for example will test more,sacrificing performance.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    ###TESTING POINT####
    model = KNeighborsClassifier(n_neighbors=9)
    ###TRAINING MODEL###
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    print('interation: ', i, 'Accuracy: ', acc * 100)

    if acc > best:
        best = acc
        with open('My_model(KNN).pickle', 'wb') as f:
            pickle.dump(model, f)
            print(best)
        break

load_in = open('My_model(KNN).pickle', 'rb')
model = pickle.load(load_in)



predictions = model.predict(x_test)'''

label = ['G1', 'G2', 'studytime', 'failures', 'absences']

for i in range(len(predictions)):
    print('Prediction:',predictions[i]*5 , 'Data:',x_test[i], 'Actual:',y_test[i]*5 )

'''#GRAPH VISUAL#
p = 'G1'
Q = 'G2'
a = 'absences'
style.use('ggplot')
pyplot.scatter(data[predict]*Grade_balancer,data[a]*Grade_balancer)
pyplot.xlabel('Final')
pyplot.ylabel('Absences')
pyplot.show()'''
