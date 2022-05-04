import numpy as np
from XGBoost import XGBClassifier
from argparse import ArgumentParser

def create_argument_parser():
    parser = ArgumentParser()
    
    parser.add_argument('X_train')
    parser.add_argument('y_train')
    parser.add_argument('X_test')
    parser.add_argument('save_dir')
    parser.add_argument('-n', type=int, default=75)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--option', choices=['raw', 'prob', 'label'], default='prob')
    parser.add_argument('--min_split', type=int, default=2)
    parser.add_argument('-d', '--depth', type=int, default=6)
    parser.add_argument('-l', '--lambda_', type=float, default=1.)
    
    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    with open(f'{args.X_train}') as f:
        ncols = len(f.readline().split(','))

    X_train = np.loadtxt(f'{args.X_train}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    X_test = np.loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    y_train = np.load(f'{args.y_train}')

    xgb_model = XGBClassifier(args.n, args.eta,args.min_split, args.depth, args.lambda_)
    xgb_model = xgb_model.fit(X_train, y_train)

    if args.option == 'raw':
        y_pred = xgb_model.predict_raw(X_test)
    elif args.option == 'prob':
        y_pred = xgb_model.predict_proba(X_test)
    else:
        y_pred = xgb_model.predict(X_test)

    np.savetxt(f'{args.save_dir}', y_pred, fmt='%f', delimiter=',')
    print(f'predictions saved to {args.save_dir}')
