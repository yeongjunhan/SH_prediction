import argparse
import pandas as pd
import numpy as np
import os, sys
shinhan_path = ""
sys.path.append(shinhan_path)
from datetime import datetime
from sklearn.decomposition import PCA
import tensorflow as tf
import main as model
import matplotlib
import matplotlib.pyplot as plt
import logging
matplotlib.use('agg')
plt.switch_backend('agg')


def preprocessing(eev_path, matket_data_path, market_name, U_l_save_path, save_path, n1, n2, n3, data_type,
                  test_start_date, pred_period, long_term=0, U_l_needed=False, seed=0):
    """
    **preprocessing func.

        Make EEV into U_l vectors(if needed) and split train / validation set, before train the model.

        :param str eev_path: EEV path
        :param str market_data_path: target_df.csv path
        :param str market_name: market_name
        :param str U_l_save_path: save_path for U_l data
        :param str save_path: save_path
        :param int n1, n2, n3: The number of each validation sets.
                               n1 means the number of 0-40% up ratio interval
                               n2 means the number of 40-60% up ratio interval
                               n3 means the number of 60-100% up ratio interval
        :param str data_type: One of '3Q', 'Ran', 'PC2'
        :param int long_term: We make the data U_l before 22 days from present
        :param bool U_l_needed: Making summarized EEV takes long time. To shorten the running time, only 1st step is
                                used if the boolean gets "True". Then later step, we just load U_l as we saved


        :return list[numpy.array]: lagged U_l. THis is U_l[pred_period:], Y_one_hot. Making True Y into one-hot encode,
                                   indices of training set, validation set, true_y, time array
    """
    if U_l_needed:

        print("Reading EEV")
        eev = pd.read_csv(eev_path)[::-1].set_index('timestamp')

        eev_result = []
        eev_res = []
        unique_index = eev.index.unique()
        print("Making Daily Vectors")

        if data_type == '3Q':
            for i, t in enumerate(unique_index):
                if (i+1) % 100 == 0:
                    print(i+1, datetime.now())

                eev_per_1 = np.percentile(eev.loc[t], 25, axis=0, interpolation='nearest')
                eev_per_2 = np.percentile(eev.loc[t], 50, axis=0, interpolation='nearest')
                eev_per_3 = np.percentile(eev.loc[t], 75, axis=0, interpolation='nearest')

                eev_result = np.concatenate((eev_per_1, eev_per_2, eev_per_3), axis=0)
                eev_res.append(np.array(eev_result))
            eev_res = pd.DataFrame(eev_res)
            eev_res.index = unique_index

            U_l = []
            n_samples = len(eev_res) - long_term + 1 # shorten lenght (lagging)
            for i in range(n_samples):
                u_l = np.array(eev_res.iloc[i:i+long_term])
                U_l.append(u_l)
            U_l = np.array(U_l)
            U_l_timestamps = pd.to_datetime(eev_res.index[:n_samples], format='%Y-%m-%d')

            np.save(U_l_save_path + '/U_l_3Q.npy', U_l)
            np.save(U_l_save_path + '/U_l_timestamps.npy', U_l_timestamps)

        elif data_type == 'Ran':
            for i, t in enumerate(unique_index):
                if (i+1) % 100 == 0:
                    print(i+1, datetime.now())

                eev_per_1 = np.percentile(eev.loc[t], 25, axis=0, interpolation='nearest')
                eev_per_2 = np.percentile(eev.loc[t], 50, axis=0, interpolation='nearest')
                eev_per_3 = np.percentile(eev.loc[t], 75, axis=0, interpolation='nearest')
                eev_div = eev_per_3 - eev_per_1

                n_comp = 1
                pca = PCA(n_components=n_comp)
                pca.fit(eev.loc[t])
                eev_pc = pca.components_.reshape((1, -1)).squeeze()

                eev_result = np.concatenate((eev_pc, eev_per_2, eev_div))
                eev_res.append(np.array(eev_result))
            eev_res = pd.DataFrame(eev_res)
            eev_res.index = unique_index

            U_l = []
            n_samples = len(eev_res) - long_term + 1 # shorten lenght (lagging)
            for i in range(n_samples):
                u_l = np.array(eev_res.iloc[i:i+long_term])
                U_l.append(u_l)
            U_l = np.array(U_l)
            U_l_timestamps = pd.to_datetime(eev_res.index[:n_samples], format='%Y-%m-%d')

            np.save(U_l_save_path + '/U_l_Ran.npy', U_l)
            np.save(U_l_save_path + '/U_l_timestamps.npy', U_l_timestamps)


        elif data_type == 'PC2':
            for i, t in enumerate(unique_index):
                if (i + 1) % 100 == 0:
                    print(i + 1, datetime.now())

                eev_per_1 = np.percentile(eev.loc[t], 25, axis=0, interpolation='nearest')
                eev_per_2 = np.percentile(eev.loc[t], 50, axis=0, interpolation='nearest')
                eev_per_3 = np.percentile(eev.loc[t], 75, axis=0, interpolation='nearest')
                eev_div = eev_per_3 - eev_per_1

                n_comp = 2
                pca = PCA(n_components=n_comp)
                pca.fit(eev.loc[t])
                eev_pc = pca.components_.reshape((1, -1)).squeeze()

                eev_result = np.concatenate((eev_pc, eev_per_2, eev_div))
                eev_res.append(np.array(eev_result))
            eev_res = pd.DataFrame(eev_res)
            eev_res.index = unique_index

            U_l = []
            n_samples = len(eev_res) - long_term + 1  # shorten lenght (lagging)
            for i in range(n_samples):
                u_l = np.array(eev_res.iloc[i:i + long_term])
                U_l.append(u_l)
            U_l = np.array(U_l)
            U_l_timestamps = pd.to_datetime(eev_res.index[:n_samples], format='%Y-%m-%d')

            np.save(U_l_save_path + '/U_l_Ran.npy', U_l)
            np.save(U_l_save_path + '/U_l_timestamps.npy', U_l_timestamps)
    else:
        U_l = np.load(U_l_save_path + '/U_l_{}.npy'.format(data_type))
        U_l_timestamps = np.load(U_l_save_path + '/U_l_timestamps.npy')

    market = pd.read_csv(matket_data_path, use_cols=['TradeDate', market_name])[::-1]
    target_market = market.rename(columns={'TradeDate': 'timestamp', market_name: 'Close'})

    # Data Split
    Y = target_market.copy().dropna()
    Y.timestamp = pd.to_datetime(Y.timestamp, format = '%Y-%m-%d')

    int_times = np.intersect1d(U_l_timestamps, Y.timestamps)[::-1]
    U_l = U_l[[True if (t in int_times) else False for t in U_l_timestamps]]
    Y = Y.loc[int_times]
    if len(Y) != len(U_l):
        print("The two are different")

    Y['label'] = Y['Close'].diff(-pred_period) > 0

    Y = Y[:-pred_period]
    U_l_lag = U_l[pred_period:]

    timestamps = Y.index.copy()
    labels = Y['label'].copy().values

    updown = [1 if v else 0 for v in list(labels)]
    Y_one_hot = np.eye(2)[updown]


    test_start_ind = np.where(timestamps == test_start_date)[0][0]
    train_last_ind = test_start_ind + pred_period

    trva_ind = range(train_last_ind, len(U_l_lag))
    trva_Y = Y.iloc[train_last_ind:]
    n_trva = len(trva_Y)
    temp = trva_Y['label'].copy()
    temp.index = range(len(temp))
    val_rad = 10
    n_val = val_rad*2 + 1
    up_ratio = temp.rolling(window=n_val, center=True).mean()
    low_ind = np.where(up_ratio < 0.4)[0]
    high_ind = np.where(up_ratio > 0.6)[0]
    mid_ind = np.setdiff1d(range(val_rad, n_trva - val_rad), np.concatenate([low_ind, high_ind]))

    np.random.seed(seed)

    i_low = np.random.choice(low_ind, size=n1, replace=False)
    i_mid = np.random.choice(mid_ind, size=n2, replace=False)
    i_high = np.random.choice(high_ind, size=n3, replace=False)
    val_ind_low_list = []
    val_ind_mid_list = []
    val_ind_high_list = []

    for n in range(n1):
        val_ind_low_list.append(np.array(trva_ind)[range(i_low[n] - val_rad, i_low[n] + val_rad + 1)].tolist())
    val_ind_low_list = np.concatenate(val_ind_low_list)

    for n in range(n2):
        val_ind_mid_list.append(np.array(trva_ind)[range(i_mid[n] - val_rad, i_mid[n] + val_rad + 1)].tolist())
    val_ind_mid_list = np.concatenate(val_ind_mid_list)

    for n in range(n3):
        val_ind_high_list.append(np.array(trva_ind)[range(i_high[n] - val_rad, i_high[n] + val_rad + 1)].tolist())
    val_ind_high_list = np.concatenate(val_ind_high_list)

    val_ind_list = [val_ind_low_list, val_ind_mid_list, val_ind_high_list]
    train_ind = np.setdiff1d(trva_ind, np.concatenate(val_ind_list))
    train_ind = np.sort(train_ind)

    return U_l_lag, Y_one_hot, train_ind, val_ind_list, Y, timestamps

def my_logger(logger_name, log_save_path, file_names):
    """
    Make log file
    :param logger_name:
    :param log_save_path:
    :param file_names:
    :return: logger
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_save_path + '/' + file_names + '.log')
    formatter = logging.formater('%(asctime)s %(levelnames)s %(message)s ')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # for console print
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger

thres_list=[]

def tuning(U_l_lag, Y_one_hot, train_ind, val_ind_list, save_path, data_type, max_iter=5):
    """
    Tuning Fn.
    :param U_l_lag: lagged U_l
    :param Y_one_hot: one hot encoded y
    :param train_ind: train index
    :param val_ind_list: validation index list
    :param save_path:
    :param data_type: One of '3Q' ,'Ran', or 'PC2'
    :param max_iter: max iteration for random search
    :return: tuning result csv file
    """

    params = {'hidden_dim': hidden_dim_list,
              'lambda_value': lambda_value_list,
              'kprob': kprob_list,
              'lr': lr_list}

    current_params = {'hidden_dim': hidden_dim_list[0],
              'lambda_value': lambda_value_list[0],
              'kprob': kprob_list[0],
              'lr': lr_list[0]}

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    try:
        result = pd.read_csv(save_path + '/result_{}.csv'.format(data_type))
    except:
        result = pd.DataFrame(columns=['hidden_dim', 'lambda_value', 'kprob', 'lr', 'data_type', 'val_acc',
                                       'thres', 'val_acc_1', 'val_acc_2', 'val_acc_3'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = my_logger('shinhan', save_path, 'training_log')

    iter = 0
    while(1):
        if iter == max_iter:
            break
        for i, param in enumerate(list(params.keys())):
            try:
                current_params[param] = result.iloc[np.argmax(result['val_acc'])][param]
                temp_params = current_params.copy()
            except:
                temp_params = current_params.copy()
            for p in params[param]:
                temp_params[param] = p
                same_params = \
                np.where((result[list(temp_params.keys())].values == list(temp_params.values())).all(axis=1))[0]
                if same_params.tolist() != []:
                    pass
                else:
                    NTNCNN_graph = tf.Graph()
                    with tf.Session(config=config, graph=NTNCNN_graph) as session:
                        with tf.device("/gpu:{}".format(gpu)):
                            NTN_CNN = model.NTNCNN(n_class=2,
                                                   hidden_dim=temp_params['hidden_dim'],
                                                   window_size=3,
                                                   kprob=temp_params['kprob'],
                                                   batch_size=256,
                                                   epoch=200,
                                                   lr=temp_params['lr'],
                                                   lambda_value=temp_params['lambda_value'],
                                                   session=session,
                                                   logger=logger,
                                                   thres=0.5,
                                                   p_w=1.,
                                                   n_w=1.,
                                                   data_type=data_type)
                            val_ind_tot = np.concatenate(val_ind_list)
                            NTN_CNN.train(U_l_lag[train_ind], Y_one_hot[train_ind], U_l_lag[val_ind_tot],
                                          Y_one_hot[val_ind_tot])
                            Y_pred_val = NTN_CNN.predict(U_l_lag[val_ind_tot])[:, 1]
                            True_y_val = Y_one_hot[val_ind_tot][:, 1]

                            val_acc_list_thres = np.empty(len(thres_list))
                            for i, thres in enumerate(thres_list):
                                _, val_acc = confusion_matrix(pred_Y=Y_pred_val, true_Y=True_y_val, thres=thres)
                                val_acc_list_thres[i] = val_acc
                            best_thres = thres_list[np.argmax(val_acc_list_thres)]
                            max_val_acc = max(val_acc_list_thres)

                            for i, val_ind in enumerate(val_ind_list):
                                Y_pred_val_s = NTN_CNN.predict(U_l_lag[val_ind])[:, 1]
                                True_Y_val_s = Y_one_hot[val_ind][:, 1]
                                _, val_acc = confusion_matrix(Y_pred_val_s, True_Y_val_s, thres=best_thres)
                                temp_params.update({'val_acc_{}'.format(i+1): val_acc})

                            last_epoch = NTN_CNN.history['last_epoch']
                            temp_params.update({'last_epoch': last_epoch})
                            temp_params.update({'thres': best_thres})
                            temp_params.update({'val_acc': max_val_acc})
                            temp_params.update({'data_type': data_type})
                            current_result = pd.DataFrame([temp_params])
                            result = result.append(current_result).reset_index(drop=True)
                            current_params[param] = result.iloc[np.argmax(result['val_acc'])][param]
        iter +=1
    result.to_csv(save_path + '/result_{}.csv'.format(data_type), index=False)
    return

def confusion_matrix(pred_Y, true_Y, thres):
    """

    :param pred_Y:
    :param true_Y:
    :param thres: cutoff value
    :return: TP TN FP FN
    """
    TN, FP, FN, TP = 0, 0, 0, 0
    pred_Y = pred_Y > thres
    for i in range(len(pred_Y)):
        if pred_Y[i] == true_Y[i]:
            TP += 1
        elif pred_Y[i] == 1 and pred_Y[i] != true_Y[i]:
            FP += 1
        elif pred_Y[i] == 0 and pred_Y[i] != true_Y[i]:
            FN += 1
        else:
            TN += 1
    conf_mat = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP}
    accuracy = (TP + TN)/(TP+TN+FN+FP)
    return conf_mat, accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action='append', help='target market list', required=True)
    parser.add_argument('-l2','--list2', action='append', help='prediction period list',required=True)
    parser.add_argument('-r', '--rand_seed', type=int, default=2018)
    parser.add_argument('-p', '--save_path', type=str, default='sda')
    args = parser.parse_args()

    market_name_list = args.list
    pred_period_list = args.list2
    pred_period_list = [int(period) for period in pred_period_list]

    market_data_path = ''
    eev_path = ''
    U_l_save_path = ''
    out_path = ''
    data_type_list = ['3Q', 'Ran', 'PC2']

    n1 = 0
    n2 = 0
    n3 = 0
    prediction = pd.read_csv(out_path + '/.csv')
    test_date = prediction['TradeDate'].iloc[-1]
    lambda_value_list = []
    hidden_dim_list = []
    kprob_list = np.arange()
    max_iter = 0
    max_epoch = 0
    lr_list = [0.]
    thres_list = np.arange()
    gpu = 1

    # Tuning step
    for i, pred_period in enumerate(pred_period_list):
        for j, market_name in enumerate(market_name_list):
            test_start_date = test_date
            save_path = U_l_save_path + '/pred{}'.format(pred_period)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for data_type in data_type_list:
                # U_l is needed
                if i==0 and j==0:
                    U_l_lag, Y_one_hot, train_ind, val_ind_list, Y, timestamps = preprocessing(**args)
                # U_l is not needed
                else:
                    U_l_lag, Y_one_hot, train_ind, val_ind_list, Y, timestamps = preprocessing(**args)

                tuning(U_l_lag, Y_one_hot, train_ind, val_ind_list, save_path, data_type, max_iter=max_iter)

            result_3Q = pd.read_csv(save_path + '/result_3Q.csv')
            result_Ran = pd.read_csv(save_path + '/result_Ran.csv')
            result_PC2 = pd.read_csv(save_path + '/result_PC2.csv')
            result = pd.concat([result_3Q, result_PC2, result_Ran])
            result.to_csv(save_path + '/result.csv', index=None)
