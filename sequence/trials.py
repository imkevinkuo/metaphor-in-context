def run_trials(dataset, n):
    # All datasets: using sequence labeling to generate verb classification
    seq_on_verb_results = [[] for i in range(5)]
    # VUA: data breakdown by POS for sequence labeling
    vua_confusion_matrices = []
    seq_on_allpos_results = [[] for i in range(5)]
    pos2idx = None

    if dataset == "trofi":
        import main_trofi

        for t in range(n):
            results = main_trofi.train_model()
            for i in range(len(results)):
                seq_on_verb_results[i].append(results[i])

    elif dataset == "mohx":
        import main_mohx

        for t in range(n):
            results = main_mohx.train_model()
            for i in range(len(results)):
                seq_on_verb_results[i].append(results[i])

    elif dataset == "vua":
        import main_vua

        for t in range(n):
            rnn_clf, nll_criterion = main_vua.train_model()
            confusion_matrix, results_verb, results_pos, pos2idx = main_vua.test_model(rnn_clf, nll_criterion)
            vua_confusion_matrices.append(confusion_matrix)
            for i in range(len(results_verb)):
                seq_on_verb_results[i].append(results_verb[i])
            for i in range(len(results_pos)):
                seq_on_allpos_results[i].append(results_pos[i])

    return seq_on_verb_results, vua_confusion_matrices, seq_on_allpos_results, pos2idx


def print_table(results, rows, cols):
    for t in range(rows):
        s = ""
        for i in range(cols):
            s += str(results[i][t]) + "\t"
        print(s)


def print_results_from_confusion_matrices(vua_confusion_matrices, trials, cols, pos2idx):
    for pos in pos2idx:
        print(pos)
        for t in range(trials):
            s = ""
            for col in range(cols):
                s += str(vua_confusion_matrices[t][pos2idx[pos]][col]) + '\t'
            print(s)
