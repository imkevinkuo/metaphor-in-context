def run_trials(dataset, n):
    all_results = [[] for i in range(5)]

    if dataset == "trofi":
        import main_trofi

        for t in range(n):
            results = main_trofi.train_model()
            for i in range(len(results)):
                all_results[i].append(results[i])

    elif dataset == "mohx":
        import main_mohx

        for t in range(n):
            results = main_mohx.train_model()
            for i in range(len(results)):
                all_results[i].append(results[i])

    elif dataset == "vua":
        import main_vua

        for t in range(n):
            rnn_clf, nll_criterion = main_vua.train_model()
            results = main_vua.test_model(rnn_clf, nll_criterion)
            for i in range(len(results)):
                all_results[i].append(results[i])

    return all_results


def print_table(results, trials, cols):
    # Precision, Recall, F1, Accuracy, MaF1 (for VUA)
    for t in range(trials):
        s = ""
        for i in range(cols):
            s += str(results[i][t]) + "\t"
        print(s)
