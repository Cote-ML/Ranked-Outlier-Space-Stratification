import numpy as np
import generic_numpy_funcs


class InfluenceSpace(object):
    '''
    Builds influence space parameters for stratification purposes. 
    TODO: Expand user guide on this. 
    '''
    def __init__(self, distance_metric='euclidean', n_neighbors=30): #Heuristically, n_neighbors should be ~1% of data size. 
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors
        self.standardization_eps = 0.5

    def generate_scores(self, data):
        self._initalize_model(data)
        dist_mat = squareform(pdist(data, metric=self.distance_metric))
        is_params = self._get_influence_space(dist_mat)
        stratification_params = self._get_stratification_params(is_params)
        ordered_strat_params = generic_numpy_funcs.sort_by_position(stratification_params, 1)
        return ordered_strat_params

    def _get_object_influence(self, dist_mat, idx):
        k_nearest_idx = np.argpartition(dist_mat[:, idx], range(self.n_neighbors + 1))[1:self.n_neighbors + 1]
        k_nearest_dist = bn.partition(dist_mat[:, idx], self.n_neighbors)
        local_density = 1 / k_nearest_dist[self.n_neighbors]
        knn_dist = sum(k_nearest_dist[1:self.n_neighbors + 1])
        return np.array([k_nearest_idx, local_density, knn_dist], dtype=object)

    def _get_influence_space(self, dist_mat):
        """
        Computational Complexity: O(k * N) - constant-adjusted linear

        Given a nearest neighborhood of integer size k, returns an nx5 array described by,
            [0] : set of a given object's k nearest objects
            [1] : k-local density for an object, e.g. den_k(x) = 1/k_dist(x)
            [2] : Sum of distances from x to its k nearest neighbors
            [3] : Reverse Nearest Neighbor Set for x
            [4] : Influence Space of the object, defined as the union between [0] and [4]
            
        Initializing an empty matrix and loop-filling with vectorized func results is considerably faster than just calling the func N times in the loop itself. 
        """

        is_params_nested = generic_numpy_funcs.partial_vectorization_fit(self._get_object_influence, self.iterable, dist_mat)
        is_params = np.empty((self.n_samples, 5), dtype=object)
        for idx in tqdm(self.iterable):
            is_params[idx, 0], is_params[idx, 1], is_params[idx, 2] = is_params_nested[idx]
            for neighbor in is_params[idx, 0]:
                current_rnn = is_params[neighbor, 3]
                is_params[neighbor, 3] = np.array([idx]) if current_rnn is None else np.append(current_rnn, [idx])

        is_params[:, 3] = [[] if rnn is None else rnn for rnn in is_params[:, 3]]
        is_params[:, 4] = [np.union1d(is_params[idx, 0], is_params[idx, 3]) for idx in self.iterable]
        return is_params

    def _get_stratification_params(self, is_params):
        """
        Given an ndarray containing the KNN sets, local densities, RNN, and IS sets, returns an nx3 array s.t.,
        [0] : inflo score indicating likelihood of being a local outlier
        [1] : Density Function of k, a weighted score based on inflo and the summation of k nearest distances.

        Used Formulae:
        - inflo_score = summation(density of all y's for y in IS(x))/ |IS(x)| * den(x)
        - density_score = INFLO(x) + w(x), w(x) = sum of distance from x to all other objects in {Knn}
        """

        def get_density_score(params, idx):
            total_density = 0
            for obj in params[idx][4]:
                total_density += params[int(obj)][1]
            inflo_score = total_density / (len(params[idx][4]) * params[idx][1])
            density_score = inflo_score + params[idx][2]
            if inflo_score is np.nan:
                raise Exception("Neighborhood Value {} not large enough to discriminate outlier clusters effectively."
                                .format(self.n_neighbors))
            return np.array([inflo_score, density_score], dtype=object)

        density_scores = generic_numpy_funcs.partial_vectorization_fit(get_density_score, self.iterable, is_params)
        strat_params = np.zeros((self.n_samples, 2))
        for idx in self.iterable:
            strat_params[idx, 0], strat_params[idx, 1] = density_scores[idx]
        return strat_params

    def _initalize_model(self, data):
        self.n_samples = len(data)
        self.iterable = range(self.n_samples)

        if type(data) is not np.ndarray:
            raise Exception("[Error] Data must be imported as an nxm data array, not {}".format(type(data)))
        if np.isnan(data).any():
            raise Exception("ERROR] NaN values contained in data")
        if data.shape[1] > data.shape[0]:
            raise Exception("[Error] Number of columns may not exceed number of records: {} vs {}"
                            .format(data.shape[1], data.shape[0]))
        if self.n_neighbors > data.shape[0]:
            raise Exception("[Error] Neighborhood range greater than the number of samples in the data.")
        if type(self.n_neighbors) is not int:
            raise Exception("[Error] Neighborhood must be a discrete integer, not {}".format(type(self.n_neighbors)))
        #if not (0 - self.standardization_eps <= data.mean() <= 0 + self.standardization_eps):
        #    raise Exception("[Error] Your data does not look standardized (mean 0, var 1). True Mean {}, True Var {}"
        #                    .format(data.mean(), data.var()))
        
        
class ROSS(object): #Ranked Outlier Space Stratification
    '''
    Ranked Outlier Space Stratification
    Takes in an ordered set of density function and (A)INFLO scores and stratifies into respective density strata 
    '''
    def __init__(self, binary_flag = False):
        self.cut = 0
        self.layer = 1
        self.gen_threshold = 0
        self.binary_flag = binary_flag

    def run(self, inflo_scores, density_scores):
        """
        Input: inflo and Density scores corresponding to underlying data, and ordered w.r.t. density_scores
        Output: Scores partitioned into underlying density-based strata.
        """
        self._initialize_function(inflo_scores, density_scores)
        self._build_threshold(inflo_scores)

        labels = np.zeros(self.n_samples)
        break_code = False
        while break_code is False:
            if self.cut > self.n_samples:
                break
            pre_cut = self.cut
            break_code = self._calc_strata(inflo_scores, density_scores)
            labels[pre_cut:self.cut] = self.layer
            self.layer += 1
        labels[self.cut:] = self.layer
        if self.binary_flag:
            labels = np.array([-1 if val == self.layer else 1 for val in labels])
            anom_percent = np.count_nonzero(labels == -1)/len(labels)
        return labels

    def _calc_strata(self, inflo_scores, density_scores):
        """
        Determines the current candidate strata's avg density and finds the cut-point when the values surpass.
        If within this new strata the avg inflo surpasses the threshold, end and call the rest anomalous.
        """
        avg_density = np.mean(density_scores[self.cut:])
        if len(density_scores[self.cut:]) == 1:
            return True
        new_cut = next(density[0] for density in enumerate(density_scores) if density[1] > avg_density)
        adj_inflo_avg = np.mean(inflo_scores[self.cut:new_cut])
        self.cut = new_cut
        if adj_inflo_avg < self.gen_threshold:
            return False
        return True

    def _initialize_function(self, inflo_scores, density_scores):
        self.n_samples = len(inflo_scores)

        if not self.n_samples == len(density_scores):
            raise Exception("[Error] Mismatched size of inflo and density arrays")
        for arr in [density_scores, inflo_scores]:
            if np.isnan(arr).any():
                raise Exception("[Error] NaN values contained in INFLO/Density Scores.")
            if np.array(arr <= 0).any():
                raise Exception("[Error] Negative densities/INFLO scores not permitted.")
        if not np.array(np.diff(density_scores) >= 0).all():
            raise Exception("[Error] Density scores must be ordered ascending.")

    def _build_threshold(self, inflo_scores):
        self.gen_threshold = np.mean(inflo_scores) + np.var(inflo_scores)
        if self.gen_threshold > max(inflo_scores):
            raise Exception("[Error] All points would be labeled inliers. Threshold {}, Max INFLO {}"
                            .format(self.gen_threshold, max(inflo_scores)))
                            
    def _get_classification(self, df):
        data = np.array(df[self.model_fields])
        adj_data = preprocessing.scale(data)
        densities = density.InfluenceSpace(n_neighbors=round((len(data)*.05))).generate_scores(adj_data)
        labels = unsupervised.ROSS(binary_flag=True).run(densities[:,0], densities[:,1])
        strata_scores = np.concatenate([densities, labels[:, None]], axis=1)
        re_sorted = np.argsort(strata_scores[:, 2])
        return strata_scores[re_sorted][:, 3]
