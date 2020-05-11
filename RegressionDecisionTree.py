from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from six import StringIO
import pydot
import pickle

task = "Pendulum-v0"
x = np.load("DATA/" + task + "/train_data_tree.npy")
y = np.load("DATA/" + task + "/train_label_tree.npy")
print("size=", np.size(y))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
k_con = []
score_con = []


def get_feature_name(task_name="Pendulum-v0"):
    feature_name = []
    if task == "Pendulum-v0" or "Pendulum":
        feature_name = ['cos(theta)', 'sin(theta)', 'theta_dot']
    elif task == "MountainCar":
        feature_name = ['position', 'velocity']
    elif task == "Acrobot-v1":
        feature_name = ['cos(theta1)', 'sin(theta1)', 'cos(theta2)', 'sin(theta2)', 'thetaDot1', 'thetaDot2']
    return feature_name


for k in range(1, 15):
    clf = DecisionTreeRegressor(max_depth=k)
    clf.fit(X_train, y_train)
    print('--------------------'
          '--------------------')
    print('label', y_test)
    print('predict', clf.predict(X_test))
    score = cross_val_score(clf, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    print(score.mean())
    k_con.append(k)
    score_con.append(score.mean())
    dot_data = StringIO()
    feature_name = get_feature_name(task)
    export_graphviz(clf, out_file=dot_data, max_depth=None,
                    feature_names=feature_name,
                    filled=True, leaves_parallel=True, impurity=True,
                    node_ids=True, proportion=False, rotate=False,
                    rounded=True, special_characters=False, precision=3)
    p = pydot.graph_from_dot_data(dot_data.getvalue())
    strr = 'Visualization/' + task + '/' + str(k) + '.pdf'
    p[0].write_pdf(strr)
    s = pickle.dumps(clf)
    f = open('decision_model/'+'dt_' + task + str(k) + '.txt', 'wb')
    f.write(s)
    f.close()
plt.plot(k_con, score_con)
plt.xlabel('Tree_depth')
plt.ylabel('neg_mean_squared_error')
plt.title(task)
stt = "experiment_image/" + task + "分析图.png"
plt.savefig(stt)
