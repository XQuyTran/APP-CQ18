from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from time import time
from glob import glob
from tqdm import tqdm
import numpy as np
import warnings

from XGB_par_v0 import predict_proba_kernel
from XGB_par_v1 import predict_proba_kernel1

from feature_extract import getFigureForImage
from feature_extract_v1 import getFigureForImage3, compare_gray



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
    parser.add_argument('-b', '--blockSize', type=int, default=0)
    
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
        print('')
        image_features = [getFigureForImage3(path) for path in tqdm(test_ls, 'self implement')]
        X_test = np.vstack(image_features)
    
        # rút trích đặc trưng - sử dụng thư viện
        cv_image_features = [getFigureForImage(path) for path in tqdm(test_ls, 'opencv')]
        cv_X_test = np.vstack(image_features)

        print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')
    elif args.image is not None:
        print('Feature extraction')
        
        start = time()
        X_test = getFigureForImage3(args.image)
        print(f'self implement: {(time() - start) * 1000}ms')

        start = time()
        cv_X_test = getFigureForImage(args.image)
        print(f'opencv: {(time() - start) * 1000}ms')
        
        print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')

        compare_gray(args.image)
        
        X_test = np.atleast_2d(X_test)
    elif args.npy is not None:
        X_test = np.load(args.npy)
    else:
        exit(2)

    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)
    
    # đọc mô hình và dự đoán - thư viện xgboost
    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)

    xgblib_model.load_model(args.learner_path)    
    xgblib_pred = xgblib_model.predict(X_test)

    # đọc mô hình và dự đoán - tự cài đặt
    print( 'XGBoost prediction')
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        start = time()
        xgb_pred_1 = predict_proba_kernel(X_test, features, values, n_classes, args.blockSize)
        print(f'prediction (parallel_v0) {(time() - start)*1000:f}ms')
        print(f'error with xgboost library classifier: {np.count_nonzero(np.argmax(xgb_pred_1, 1) - xgblib_pred)}', '\n')

        start = time()
        xgb_pred_2 = predict_proba_kernel1(X_test, features, values, n_classes, args.blockSize)
        print(f'prediction (parallel_v1) {(time() - start)*1000:f}ms')
        print(f'error with xgboost library classifier: {np.count_nonzero(np.argmax(xgb_pred_2, 1) - xgblib_pred)}')
    

    # print("classes: ['healthy', 'multiple_diseases', 'rust', 'scab']")
    # print(f'pred: \t\t{xgb_pred}')
    # if args.blockSize > 0:
    #     print(f'pred1: \t\t{xgb_pred_1}')
    # print(f'pred (library): {xgblib_pred}')
