import numpy as np


class Clustering:
    def __init__(self):
        self.Cnt = None
        self.S = None
    
    def _distance(self, a, b):
        return np.linalg.norm(a - b)
    
    def _affinity(self, a, b, sigma):
        d = self._distance(a, b)
        return np.exp(-d**2 / (2 * sigma))
    
    def _add_point(self, k, z_j):
        self.Cnt[k] = (self.S[k] * self.Cnt[k] + z_j) / (self.S[k] + 1)
        self.S[k] += 1
        return
    
    def _remove_point(self, z_j, c_j):
        k_prime = c_j
        if self.S[k_prime] == 1:
            self.Cnt[k_prime] = 0
            self.S[k_prime] = 0
        else:
            self.Cnt[k_prime] = (self.S[k_prime] * self.Cnt[k_prime] - z_j) / (self.S[k_prime] - 1)
            self.S[k_prime] -= 1
        return
    
    def _sort_cluster_numbers(self, C):
        # sort clusters by their size in descending order
        cluster_sizes = []
        for i in range(1, len(self.S)):
            cs = self.S[i]
            cluster_sizes.append((i, cs))
        cluster_sizes = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
        
        cluster_old_to_new = {}
        cluster_new_to_old = {}
        for i in range(1, len(self.S)):
            new = i
            old = cluster_sizes[i - 1][0]
            cluster_old_to_new[old] = new
            cluster_new_to_old[new] = old
        
        # change the order of clusters in self.S, self.Cnt and C
        S_copy = self.S.copy()
        Cnt_copy = self.Cnt.copy()
        for i in range(len(C)):
            old = C[i]
            if old != 0:
                new = cluster_old_to_new[old]
                C[i] = new
        for old_cluster_no in cluster_old_to_new.keys():
            new_cluster_no = cluster_old_to_new[old_cluster_no]
            self.S[new_cluster_no] = S_copy[old_cluster_no]
            self.Cnt[new_cluster_no] = Cnt_copy[old_cluster_no]
        del S_copy, Cnt_copy
        debug = (cluster_sizes, cluster_new_to_old, cluster_old_to_new)
        return C, debug

    def _find_clusters(self, X, remove_outliers=False):
        n, d = X.shape
        
        # normalise X
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        Z = (X - mu[None, :]) / sigma[None, :]
        
        # distance matrix
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = self._distance(Z[i], Z[j])
                #D[i, j] = np.linalg.norm(Z[i] - Z[j])

        # affinity matrix and function
        sigma_D = np.std(D)
        A = np.exp( -D ** 2 / (2 * sigma_D))
        # del D
        
        # histogram
        H = np.histogram(A, bins=10, range=(0, 1))[0]
        
        # affinity threshold
        k = np.argmax(np.diff(H)) + 1
        threshold = (k - 0.5) / 10
        
        # cluster no. of point C, size of cluster S, centroid of cluster Cnt
        C = np.zeros(n, dtype=int)
        self.S = np.zeros(n + 1, dtype=int)
        self.Cnt = np.zeros((n + 1, d), dtype=float)
        k = 0
        for i in range(n):
            if C[i] == 0:
                k += 1
                C[i] = k
                self.S[k] = 1
                self.Cnt[k] = Z[i]
            for j in range(n):
                if C[j] == 0:
                    if self._affinity(self.Cnt[k], Z[j], sigma_D) > threshold:
                        self._add_point(k, Z[j])
                        C[j] = k
                elif self._distance(self.Cnt[k], Z[j]) < self._distance(self.Cnt[C[j]], Z[j]):
                    self._remove_point(Z[j], C[j])
                    self._add_point(k, Z[j])
                    C[j] = k
        
        # remove outliers and count no. of clusters
        p = len(set(C))
        if remove_outliers:
            for i in range(n):
                if self.S[C[i]] == 1:
                    c_i = C[i]
                    self.S[c_i] = 0
                    self.Cnt[c_i] = 0
                    C[i] = 0
                    p -= 1
        
        # change cluster order
        C, debug = self._sort_cluster_numbers(C)
        
        return Z, C, p, debug
        
        
    def _merge_clusters(self, Z, C, p, verbose=False):
        n, d = Z.shape
        
        # find no. of clusters k which satisfies condition
        k = 0
        for k in range(2, p + 1):
            t1 = 0
            for i in range(1, k):
                t1 += (self.S[i] - self.S[k]) * self.S[i]
            t2 = 0
            for i in range(k + 1, p + 1):
                t2 += (self.S[k] - self.S[i]) * self.S[i]
            if t1 > t2:
                break
        
        # merge
        C_2 = C.copy()
        S_2 = self.S.copy()
        Cnt_2 = self.Cnt.copy()
        
        '''
        MERGE CLUSTERS
        0. calculate table of distances bewteen clusters' centroids
        1. find 2 closests clusters (given the precalculated table of distances between their centroids)
        2. reassign elements to the one with the row id number, changle the size
        3. recalculate the centroid
        4. recalculate the distance from the new cluster to the remaining ones
        5. go to 1
        '''
        # 0) Cnt_2 distances
        Cnt_D = np.full(fill_value=np.inf, shape=(p + 1, p + 1), dtype=float)
        for i in range(1, p + 1):
            for j in range(1, p + 1):
                Cnt_D[i, j] = self._distance(Cnt_2[i], Cnt_2[j])
            Cnt_D[i, i] = np.inf
        
        if verbose:
            print('at first,', len(set(C_2)), 'clusters') 
            print(p - k, 'merges')
        for i in range(p - k):
            # TODO: merge 2 closest clusters, update C_2
            # 1) find 2 closest clusters
            idx = np.argmin(Cnt_D)
            row, col = idx // (p + 1), idx % (p + 1)

            # 2-3) reassign elements and recalculate centroid
            Cnt_2[col] = 0
            S_2[col] = 0
            
            Cnt_2[row] *= S_2[row]
            for j in range(n):
                if C_2[j] == col:
                    C_2[j] = row
                    Cnt_2[row] += Z[j]
                    S_2[row] += 1
            Cnt_2[row] /= S_2[row]

            # 4) recalculate the distance
            for j in range(1, p + 1):
                Cnt_D[j, col] = Cnt_D[col, j] = np.inf
            for j in range(1, p + 1):
                if Cnt_D[j, row] != np.inf:
                    Cnt_D[j, row] = Cnt_D[row, j] = self._distance(Cnt_2[j], Cnt_2[row])
        
        
        # calculate metric W
        def metric_w(Z, S, C, Cnt):
            res = 0
            for j in range(1, len(S)):
                if S[j]:
                    res_j = 0
                    for i in range(n):
                        if C[i] == j:
                            res_j += np.linalg.norm(Z[i] - Cnt[j]) ** 2
                    res_j /= S[j]
                    res += res_j
                    j += 1
            return res
        W_C = metric_w(Z, self.S, C, self.Cnt)
        W_C_2 = metric_w(Z, S_2, C_2, Cnt_2)
        if W_C_2 > W_C:
            # discard merging
            C, _ = self._sort_cluster_numbers(C)
            return C, p, (W_C, W_C_2)
        self.Cnt = Cnt_2
        self.S = S_2
        C_2, _ = self._sort_cluster_numbers(C_2)
        return C_2, k, (W_C, W_C_2)
                
    
    def fit_predict(self, X, remove_outliers=False, verbose=False):
        Z, C, p, debug = self._find_clusters(X, remove_outliers)
        if verbose:
            print('before', p, len(set(C)))
        C, p, (W_1, W_2) = self._merge_clusters(Z, C, p, verbose)
        if verbose:
            print(W_1, '->', W_2)
            print('after', p, len(set(C)), '\n---------------------------------\n')
        return C, p
