import numpy as np
from sklearn.model_selection import train_test_split
from hyppo.ksample import Energy
from tsp import gibb_sampl_sampling
from dataset_load import dataset_loader, DATASETS

np.random.seed(54321)

MAX_N = 1500
R = 10
P_MISS = 0.25

results = {}

for dataset in DATASETS:
    if dataset == 'boston':
        continue
    print(f"\n=== {dataset} ===")
    try:
        X_orig = dataset_loader(dataset)
        n_full, d = X_orig.shape
        if n_full > MAX_N:
            idx = np.random.choice(n_full, MAX_N, replace=False)
            X_orig = X_orig[idx]
        n, d = X_orig.shape

        mean = np.mean(X_orig, axis=0)
        std = np.std(X_orig, axis=0)
        std[std == 0] = 1.0
        X = (X_orig - mean) / std

        M = np.random.binomial(n=1, p=P_MISS, size=(n, d))
        X_train, X_test, M_train, M_test = train_test_split(X, M, test_size=0.30)
        n_tr = X_train.shape[0]

        info_dic = {
            'data': X_train,
            'masks': M_train,
            'imputed_data': None,
            'nbr_it_gibb_sampl': 1,
            'lbd_reg': 0.001,
            'tsp': False,
            'batch_size': 64,
            'verbose': 0,
            'initial_strategy': 'constant',
            'exponent_d': 0.75,
            'sampling': True,
            'intercept': True,
        }

        stat_arr = np.zeros(R)
        pval_arr = np.zeros(R)
        for i in range(R):
            X_train_filled = gibb_sampl_sampling(info_dic)
            info_dic['imputed_data'] = X_train_filled
            stat, pvalue = Energy().test(X_test, X_train_filled)
            stat_arr[i] = stat
            pval_arr[i] = pvalue

        slope_stat = np.polyfit(np.arange(R), stat_arr, 1)[0]
        slope_pval = np.polyfit(np.arange(R), pval_arr, 1)[0]
        good = (slope_stat < 0) and (slope_pval > 0)

        results[dataset] = dict(n=n, d=d, slope_stat=slope_stat, slope_pval=slope_pval,
                                 good=good, stat_arr=stat_arr, pval_arr=pval_arr)
        print(f"n={n} d={d} slope_stat={slope_stat:.6f} slope_pval={slope_pval:.6f} good={good}")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        results[dataset] = dict(error=str(e))

print("\n\n==== SUMMARY ====")
good_ones = []
for name, r in results.items():
    if 'error' in r:
        print(f"{name:30s} ERROR: {r['error']}")
    else:
        flag = "GOOD" if r['good'] else "    "
        print(f"{name:30s} n={r['n']:5d} d={r['d']:3d} slope_stat={r['slope_stat']:+.6f} slope_pval={r['slope_pval']:+.6f} {flag}")
        if r['good']:
            good_ones.append(name)

print("\nDatasets with decreasing energy distance AND increasing p-value:", good_ones)
