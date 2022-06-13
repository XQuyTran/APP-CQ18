from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from time import time
from glob import glob
from tqdm import tqdm
import numpy as np
from feature_extract import getFigureForImage2, getFigureForImage


def create_argument_parser():
    '''
    Hàm phân tích cú pháp tham số dòng lệnh
    '''
    parser = ArgumentParser()

    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')

    parser.add_argument('-d', help='đường dẫn tới thư mục hình đầu vào')
    parser.add_argument('-i', '--image', help='đường dẫn tới hình đầu vào')
    parser.add_argument('--npy', help='đường dẫn tới tập tin đặc trưng đã trích xuất')
    
    return parser


def get_sort_key(s):
    start = s.rfind('_') + 1
    end = s.rfind('.')
    return int(s[start:end])


if __name__ == '__main__':
    # đọc tham số dòng lệnh
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.d is not None:
        print('Feature extraction')

        test_ls = glob(f'{args.d}/Test_*')
        test_ls = sorted(test_ls, key=get_sort_key)

        # rút trích đặc trưng - tự cài đặt
        image_features = [getFigureForImage2(path) for path in tqdm(test_ls, 'self implement')]
        X_test = np.vstack(image_features)
    
        # rút trích đặc trưng - sử dụng thư viện
        cv_image_features = [getFigureForImage(path) for path in tqdm(test_ls, 'opencv')]
        cv_X_test = np.vstack(image_features)

        print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')
    elif args.image is not None:
        print('Feature extraction')

        start = time()
        X_test = getFigureForImage2(args.image)
        print(f'self implement: {(time() - start) * 1000}ms')

        start = time()
        cv_X_test = getFigureForImage(args.image)
        print(f'opencv: {(time() - start) * 1000}ms')

        print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')
        
        X_test = np.atleast_2d(X_test)
    elif args.npy is not None:
        X_test = np.load(args.npy)
    else:
        exit(2)

    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)
    
    # đọc mô hình và dự đoán - thư viện xgboost
    print('XGBoost prediction')
    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)

    xgblib_model.load_model(args.learner_path)    

    start = time()
    xgblib_pred = xgblib_model.predict(X_test)
    print(f'library: {(time() - start)*1000:f}ms')

    # đọc mô hình và dự đoán - tự cài đặt
    start = time()
    xgb_pred = np.argmax(predict_proba(X_test, features, values, n_classes), 1)
    print(f'self implement: {(time() - start)*1000:f}ms')

    print(f'error with xgboost library classifier: {np.count_nonzero(xgb_pred - xgblib_pred)}')
    