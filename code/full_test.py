from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from numpy import loadtxt
from time import time
from XGBpar import predict_proba_kernel
from glob import glob
from tqdm import tqdm
from feature_extract import getFigureForImage2, getFigureForImage
from feature_extract_v1 import getFigureForImage3, compare_gray
import numpy as np
import warnings

def create_argument_parser():
    '''
    Hàm phân tích cú pháp tham số dòng lệnh
    '''
    parser = ArgumentParser()

    parser.add_argument('image_path', help='đường dẫn tới hình')
    # parser.add_argument('X_test', help='đường dẫn tới tập dữ liệu kiểm tra')
    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')

    # parser.add_argument('-p', '--parallel', type=int, choices=[0, 1, 2, 3], default=0,
    #                                     help='''chạy tuần tự hoặc song song:
    #                                                 0 - chạy tuần tự;
    #                                                 1 - chạy song song;
    #                                                 2 - chạy song song và tối ưu hóa (v1);
    #                                                 3 - chạy song song và tối ưu hóa (v2);''')
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

    # train_ls = glob(f'{args.image_path}/Train_*')
    # train_ls = sorted(train_ls, key=get_sort_key)

    
    # rút trích đặc trưng - tự cài đặt
    # image_features = [getFigureForImage2(path) for path in tqdm(train_ls)]
    # X_train = np.vstack(image_features)

    # exit(0)
    
    # rút trích đặc trưng - sử dụng thư viện
    # cv_image_features = [getFigureForImage(path) for path in tqdm(train_ls)]
    # cv_X_train = np.vstack(image_features)

    
    #print(f'feature extracted (opencv) in {(time() - start) * 1000}ms')
    
    #print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')

    # đọc mô hình và dự đoán - tự cài đặt
    # xác định số cột đặc trưng, bỏ qua cột tên ảnh ở cuối
    # with open(f'{args.X_test}') as f:
    #     ncols = len(f.readline().split(','))

    # X_test = loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    
    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)

    print('Feature extraction')
    cv_X_test = getFigureForImage(args.image_path)

    print('Sequential')
    start = time()
    X_test = getFigureForImage2(args.image_path)
    print(f'feature extracted in {(time() - start) * 1000}ms')
    print(f'mean error: {np.abs(X_test - cv_X_test).mean()}', '\n')

    print('Parallel')
    start = time()
    X_test_1 = getFigureForImage3(args.image_path)
    print(f'feature extracted in {(time() - start) * 1000}ms')
    print(f'mean error: {np.abs(X_test_1 - cv_X_test).mean()}', '\n')

    compare_gray(args.image_path)
    
    X_test = np.atleast_2d(X_test)
        
    start = time()
    print('\n', 'XGBoost prediction')
    xgb_pred = predict_proba(X_test, features, values, n_classes)
    print(f'prediction: {(time() - start)*1000:f}ms')
    
    
    start = time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xgb_pred_1 = predict_proba_kernel(X_test, features, values, n_classes, args.blockSize)
    print(f'prediction (parallel) {(time() - start)*1000:f}ms')

    # đọc mô hình và dự đoán - thư viện xgboost
    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xgblib_model.load_model(args.learner_path)    

    xgblib_pred = xgblib_model.predict_proba(X_test)

    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    print(f'classes: {classes}')
    print(f'pred: \t\t{xgb_pred}')
    print(f'pred1: \t\t{xgb_pred_1}')
    print(f'pred (library): {xgblib_pred}')

    #print(f'mean error with xgboost library classifier: {abs(xgb_loader_pred - xgblib_pred).mean()}')
