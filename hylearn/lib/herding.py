import numpy as np

def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    # pmu = np.mean(D, axis=1)
    mu =np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))
    # print(D.shape)
    w_t = mu
    # print(w_t.shape)
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = -np.linalg.norm(w_t.reshape(1,-1)-D.T, axis=-1)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu- D[:, ind_max]
    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]

