from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from numpy import loadtxt, abs
from time import time

def create_argument_parser():
    parser = ArgumentParser()

    parser.add_argument('X_test', help='đường dẫn tới tập dữ liệu kiểm tra')
    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')
    
    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    with open(f'{args.X_test}') as f:
        ncols = len(f.readline().split(','))

    X_test = loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))

    print('Self implement loader')
    start = time()
    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)
    print(f'model loaded in {(time() - start)*1000:f}ms')

    start = time()
    xgb_loader_pred = predict_proba(X_test, features, values, n_classes)
    print(f'prediction in {(time() - start)*1000:f}ms', '\n')

    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)
    xgblib_model.load_model(args.learner_path)    
    xgblib_pred = xgblib_model.predict_proba(X_test)

    print(f'mean error with xgboost library classifier: {abs(xgb_loader_pred - xgblib_pred).mean()}')