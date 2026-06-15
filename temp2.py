def gibb_sampl_under_parametrized_sampling(info):
    # flip matrix
    #if info['ml_or_bs'] not in ['bayesian', 'max_lh']:
    #print("please specify a correct approach, bayesian or max_lh")
    #input()
    X = info['data']
    original_X = X
    M = info['masks']
    sampling = info['sampling'] if 'sampling' in info else False
    intercept = info['intercept'] if 'intercept' in info else True
    #print(M)
    #plot2D(X, M) if original_X.shape[1] == 2 else print("dimension too high, no 2D plot")
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = initial_imputation.fit_transform(X_nan) if info['imputed_data'] is None else info['imputed_data']
    #print(X)
    #plot2D(X, M) if info['plot2D_it'] else print("no 2D plot")
    #print("simple imputer in gibb sample under param \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    
    if info['tsp']:
        start_time = time.time()
        MM = M if np.mean(M) >= 1/2 else 1 - M
        M_s = csr_matrix(MM)
        ones_d = np.ones(d)
        #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
        #M_ss = csr_matrix(M)
        F = np.outer(ones_d, np.sum(MM, axis=0)) + np.outer(np.sum(MM.T, axis=1), ones_d) - 2 * M_s.T @ M_s
        #FF = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_ss.T @ M_ss
        #np.testing.assert_allclose(F, FF)
        #permutation, distance = solve_tsp_local_search(F)
        permutation, distance = serialization_first_idea(F)
        current_distance = distance
        current_permutation = permutation
        
        original_cost = np.sum(np.diag(F, k=1))
        print("original cost in tsp2", original_cost)
        #print("optimal perm ", permutation, "optimal dist ", distance) 
        distances = []
        distances.append(distance)
        s = int(np.floor(np.sqrt(d)))
        for i in range(s):
            permutation, distance = serialization_first_idea(F)
            distances.append(distance)
            if distance < current_distance:
                current_distance = distance
                current_permutation = permutation
        M = M[:, current_permutation]
        X = X[:, current_permutation]
        print("distances tsp ", np.array(distances))
        end_time = time.time()
        print(f"Execution time tsp: {end_time - start_time:.4f} seconds")

    #print("exponent d ", info['exponent_d'])
    #print("\n", X)
    #print("\n", M)
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    #print("\n ", first_mask)
    #X = X * (1/np.sqrt(n))  # normalize the column, so that the final matrix will be the covariance matrix 
    R = X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    
    #start_gibb_s = time.time()
    
    mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1])
    u = np.ones(R.shape[0])
    #print(a)
    #print("\n", np.outer(u, a))
    R_centered = R - np.outer(u, mean)
    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1/R_centered.shape[0] 
    #print("alpha ", alpha)
    Cov = R_centered.T @ R_centered + lbd * np.eye(d)  ## look how to add the (1/n), where it is better to be added 
    #start_inner1 =time.time()
    Q = np.linalg.inv(Cov)
    #end_inner1 = time.time() - start_inner1
    #print("time first inversion ", end_inner1)
    current_info ={
        'inverse': Q,
        'vectors': R
    } 
    counter_upd_dwd = 0
    counter_recomputation = 0
    counter_swm_formula = 0 
    counter_reinversion = 0
    old_X = X
    print("d ** exp: ", d ** info['exponent_d'])
    for h in range(r):
        print("\n\n CURRENT ITERATION GIBBS SANMPLING IN UNDERPARAMETRIZED SAMPLING ", h, "\n")
        old_X = X
        #diff = np.sum((X - old_X)**2)
        #stat, pvalue = Energy().test(original_X, old_X)
        #print("stat ", stat , "pvalue ", pvalue)
        #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
        #plot2D(X, M, extra_info) if info['plot2D_it'] else print("no 2D plot")
        #aa, bb = np.ones((n,)) / n, np.ones((n,)) / n
        #MM = ot.dist(original_X, old_X)
        #G0 = ot.sinkhorn2(aa, bb, MM, 0.1)
        #print("G0 ", G0)
        #input()
        #print("\ndifference old vs new \n", np.sqrt(diff))
        #plot2D(X, M) if original_X.shape[1] == 2 else print("dimension too high, no 2D plot")
        for i in range(d):
            start = time.time()
            #alpha = R_centered.shape[0] if info['ml_or_bs'] == 'bayesian' else 1
            alpha = R_centered.shape[0]  # we need to correct by this alpha to get a correct imputation
            X, _ = impute_matrix_under_parametrized_sampling(X, mean, Cov / alpha, Q * alpha, M, i, sampling, intercept)
            #print("stopppp")
            #input()
            #print("round ", i, "who is X gs\n", X)
            #v = X.T @ X[:, i]
            #Rt_R[i, :] = v
            #Rt_R[:, i] = v
            #Rt_R
            #print("who is Rt_R \n", Rt_R)
            if h < r-1 or i < d-1:
                N = Ms[:, i]
                #print("flip vector ", N)
                X_up, X_dw = split_upd(X, N)  # observe, in the fix there are also vector that were not present in neither of masks, i.e rows with (1, 1)
                #print(N)
                #print("sequence of print")
                #if info['verbose'] > 0:
                #    print(X)
                #print(X_upd)
                #print(X_dwd)
                nup, _ = X_up.shape
                #nfx, _ = X_fx.shape
                ndw, _ = X_dw.shape
                #ns = nfx + nup
                #print(nfx + nup + ndw)
                #print("ns outside", ns)
                #print("nup nfx ndw, nup", nup, ", nfx ", nfx, ", ndw ", ndw)
                '''
                if nupd + ndwd > n:
                    idx = i+1 if i<d-1 else 0
                    print(idx)
                    R = X[M[:, idx] == 0, :]
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    Rt_R = R.T @ R + lbd * np.eye(d)
                    Q = np.linalg.inv(Rt_R)
                '''
                idx = i+1 if i<d-1 else 0
                #print("idx ", idx)
                #print("nbr seen ", n - np.sum(M[:, 0]), " nbr flip ", nup + ndw)
                #print("masks \n", M[:, i:(i+2)])
                old_R = R  # old_seen components, not centered
                ns_old = old_R.shape[0]
                R = X[M[:, idx] == 0, :]  # seen components, not centered
                ns = R.shape[0]
                #print("ns true ", R.shape[0])
                old_mean = mean #if intercept else np.zeros(R.shape[1])
                mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1]) # new mean
                old_R_centered = R_centered
                u = np.ones(R.shape[0])
                R_centered = R - np.outer(u, mean)
                if ns < nup + ndw:  # if nbr seen component is less than nbr of flips
                    #print("recompute the matrix with the missing components")
                    counter_recomputation = counter_recomputation + 1
                    #mean = np.mean(R, axis=0)
                    u = np.ones(R.shape[0])
                    #print(a)
                    #print("\n", np.outer(u, a))
                    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1 / R_centered.shape[0]
                    Cov = R_centered.T @ R_centered + lbd * np.eye(d)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                else:
                    counter_upd_dwd = counter_upd_dwd + 1
                    #print("update the covariance matrix") 
                    #old_mean = mea
                    #mean = np.mean(R, axis=0)
                    old_Cov = Cov
                    Cov = Cov + np.outer(old_mean, old_mean) * ns_old + X_up.T @ X_up - X_dw.T @ X_dw - np.outer(mean, mean) * ns
                    #print("        cond numb cov in tsp ", np.linalg.cond(Cov), "it: i ", i)
                    #Cov_test = R_centered.T @ R_centered + lbd * np.eye(d)
                    #np.testing.assert_allclose(Cov_test, Cov)
                    #Cov = ((Cov - lbd * np.eye(d) + np.outer(old_mean, old_mean)) * ns_old + X_up.T @ X_up - X_dw.T @ X_dw) / ns - np.outer(mean, mean) + lbd * np.eye(d)
                if  nup + ndw > d ** info['exponent_d']:
                    #print("invert the matrix")
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd too big, invert the matrix ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    #idx = i+1 if i<d-1 else 0
                    #print(idx)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    counter_reinversion = counter_reinversion + 1
                    Q = np.linalg.inv(Cov)
                else:
                    #print("upd + dwd ", nup + ndw, " ", ns, "iteration: i ", i)
                    counter_swm_formula = counter_swm_formula + 1
                    #print("low rank upd of the inverse")
                    #print("approach: ", info['ml_or_bs'])
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd small, swm formula.          ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1 / R_centered.shape[0]
                    old_mean_rescaled = np.sqrt(ns_old) * old_mean #if info['ml_or_bs'] == 'bayesian' else old_mean 
                    mean_rescaled = np.sqrt(ns) * mean #if info['ml_or_bs'] == 'bayesian' else mean
                    #old_mean_rescaled = np.sqrt(ns_old) * old_mean 
                    #mean_rescaled = np.sqrt(ns) * mean
                    #time.sleep(1)
                    start_inner =time.time()
                    #time.sleep(2)
                    X_up_ext = np.vstack((X_up, old_mean_rescaled))
                    X_dw_ext = np.vstack((X_dw, mean_rescaled))
                   
                    #QQ = swm_formula(Q, old_mean_rescaled, 1.0)
                    #QQ_test = np.linalg.inv(old_R.T @ old_R + np.eye(d) * lbd)
                    #print("QQ_inv\n ", QQ)
                    #print("QQ_test_inv \n", QQ_test)

                    #QQQ = swm_formula(QQ, X_up.T, 1.0)
                    #QQQ = swm_formula(QQQ, X_dw.T, -1.0) 
                    #QQQ_test = np.linalg.inv(R.T @ R + np.eye(d) * lbd)
                    #print("QQQ_inv\n ", QQQ)
                    #print("QQQ_test_inv\n ", QQQ_test)

                    #QQQQ = swm_formula(QQQ, mean_rescaled, -1.0)
                    #QQQQ_test = np.linalg.inv(R_centered.T @ R_centered + np.eye(d) * lbd)
                    #print("\n\nQQQQ_inv\n ", QQQQ)
                    #print("QQQQ_test_inv \n", QQQQ_test)


                    Q = swm_formula(Q, X_up_ext.T, 1.0)
                    Q = swm_formula(Q, X_dw_ext.T, -1.0)
                    inner_elapsed = time.time() - start_inner
                    #print("inner elapsed ", inner_elapsed)
                    #Q_test = np.linalg.inv(Cov)

                    #print("Q_inv\n ", Q)
                    #print("Q_test_inv \n", Q_test)
                    #np.testing.assert_allclose(Q, Q_test)
                    #input()
                    #for i_up in range(nupd):
                    #    Q = rk_1_update_inverse(Q, X_upd[i_up, :], 1.0)
                    #for i_dw in range(ndwd):
                    #    Q = rk_1_update_inverse(Q, X_dwd[i_dw, :], -1.0)
                    #print("QQ\n ", QQ)
                    #print("Q\n", Q)
                    #print("cond nub Q in gibb sampl: ", np.linalg.cond(Q))
            end = time.time() - start
            #print("final time one iter: ", end)
    #end_gibb_s = time.time()
    #stat, pvalue = Energy().test(original_X, old_X)
    #print("stat ", stat , "pvalue ", pvalue)
    #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
    #plot2D(X, M, extra_info) if info['plot2D_it'] else print("no 2D plot")
    print("counter recomp ", counter_recomputation/r)
    print("counter upd dwd ", counter_upd_dwd/r)
    print("counter reinv", counter_reinversion/r)
    print("counter swm ", counter_swm_formula/r)
    #print("res my imp \n", X)
    start_gibb_s = 1
    end_gibb_s = 2
    print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    return X
