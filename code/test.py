# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from argparse import ArgumentParser

def create_argument_parser():
    parser = ArgumentParser()
    
    parser.add_argument('X_train')
    parser.add_argument('y_train')
    parser.add_argument('X_test')
    parser.add_argument('--progress', choices=[True, False], type=bool)
    
    return parser

if __name__ == '__main__':
    # data = load_breast_cancer()
    # X, y = data.data, data.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # model = DecisionTreeClassifier(max_depth=10)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # print('Accuracy: ', accuracy_score(y_test, y_pred))

    parser = create_argument_parser()
    args = parser.parse_args()

    with open(f'{args.X_train}') as f:
        ncols = len(f.readline().split(','))

    X_train = np.loadtxt(f'{args.X_train}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    X_test = np.loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    y_train = np.load(f'{args.y_train}')

    xgb_model = XGBClassifier(75, 0.2, progress=args.progress)

    xgb_model = xgb_model.fit(X_train, y_train)
    y_pred_prob = xgb_model.predict_proba(X_test)

    np.savetxt('predict_proba.csv', y_pred_prob, fmt='%f', delimiter=',')
