
if gridsearch==True:

    X_true=X_test #set predictors and data to use as true X
    y_true=y_test #set targets to use as true y

    eta_vals = np.logspace(-4, 1, 6)
    lmbd_vals = np.logspace(-4, 1, 6)

    train_area = np.zeros((len(eta_vals), len(lmbd_vals)))

    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()

    def NN_loop(a, b):
            NN=NeuralNetwork(X_train,y_train_onehot,n_categories=2,epochs=epochs_NN,batch_size=batch_size_NN,eta=eta_vals[a],lmbd=lmbd_vals[b])
            NN.train(plot=False)
            y_predicted_loop=NN.predict(X_true)
            area_sc,accuracy_sc = gains_area(y_true,y_predicted_loop)
            print(a,b)
            return area_sc

    results=Parallel(n_jobs=num_cores-1)(delayed(NN_loop)(a=i, b=j) for i in range(0, len(eta_vals)) for j in range(0, len(lmbd_vals)))

    print(results)
    print(eta_vals)
    print(lmbd_vals)
    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            train_area[i][j]=results[i*len(lmbd_vals)+j]

        import seaborn as sns
    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_area, annot=True, ax=ax, cmap="viridis",xticklabels=eta_vals, yticklabels=lmbd_vals)
    ax.set_title("Gains area ratio")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.savefig("../Results/gridsearch_NN.pdf")
    plt.show()
