"""
Pathing

Given a set of data points (2D), identify paths to visit each point once and determine the distance traveled. Related: traveling salesman problem.
"""

import itertools
import collections
import csv
import numpy as np
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt

__author__ = "Chris Alexiu"
__version__ = '1'
__maintainer__ = "Chris Alexiu" 
__status__ = "Production"

# =============================================================================
# 0 setup: data points, distance matrix
# -----------------------------------------------------------------------------
def data_points_generate(n=3, lo=1, hi=10, seed=None):
    """
    Generate some 2D data points with random uniform distribution.
    - Parameters:
      • n: number of data points to make, should be > 3
      • lo: minimum value
      • hi: maximum value
      • seed: randomization seed; optional
    - Return:
      • data_points: NumPy array 2D data points
    """
    if seed != None:
        np.random.seed(seed)
    data_points = np.random.uniform(low=lo, high=hi, size=(n,2))
    return data_points

def data_points_dist_mat(data_points):
    """
    Calculate distance among all pairs of data points (pairwise distances) and
    return distance matrix.
    - Parameters:
      • data: data points as nD NumPy array
    - Return:
      • pair_dist: NumPy array pairwise distance matrix
    """
    pair_dist = spatial.distance.pdist(X=data_points)
    pair_dist = spatial.distance.squareform(X=pair_dist)
    return pair_dist
# =============================================================================


# =============================================================================
# 1 generate list of paths: all or set
# -----------------------------------------------------------------------------
def paths_all_or_set_list(all_or_set="all", n_locs=3, save=1):
    """
    Generate list of possible paths to visit each data point once.
    - WARNING: quantity of possible paths may be HUGE (e.g., 20! = 2.4x10^18)
    - This function does not do any distance calculation.
    - Thus function must be supplied with a value for 'n_locs', the quantity of
      locations (data points that you will be working with - do not provide the
      actual data points).
    - Parameters:
      • all_or_set: string, "all" or "set";
        • if "all", generate all possible paths INcluding reverse duplicates
          (e.g., 1->2->3 != 3->2->1; take both), quantity of paths = n!;
        • if "set", generate all possible paths EXcluding reverse duplicates
          (e.g., 1->2->3 = 3->2->1; take one), quantity of paths = n!/2;
      • n_loc: integer; quantity of locations (data points) that you will be
        working with
      • save: 1 or string; default=1; if 1, save results to object; if string
        (filename), save results to CSV file (include the '.csv' extension)
    - Return:
      • if save=1: iterator of tuples; paths (save results to object)
      • if save=string: None (save results to CSV file)
    """
    paths_all = itertools.permutations(range(n_locs))
    # get paths (all_or_set): if 'all', use all - if 'set', filter paths to remove reverse duplicates
    if all_or_set == "all":
        paths = paths_all
    if all_or_set == "set":
        paths = set()
        for path in paths_all:
            if path not in paths and tuple(reversed(path)) not in paths:
                paths.add(path)
        paths = iter(paths)
    # save results (p_save): if 1, save to object - if string, save to CSV file
    if save == 1:
        return paths
    if type(save) == str:
        csv_file = open(save, 'w')
        csv_wrtr = csv.writer(csv_file, delimiter=',')
        for path in paths:
            csv_wrtr.writerow((path))
        csv_file.close()
        return None
# =============================================================================


# =============================================================================
# 2. calculate distance for 'all' or 'set' paths
# -----------------------------------------------------------------------------
def paths_all_or_set_strm(paths):
    """
    Produce a generator (stream) of paths from an object such as a list, tuple,
    iterator, or CSV file produced with paths_all_or_set_list().
    - This generator is meant to be used within paths_all_or_set_dist().
    - Parameters:
      • paths: list, tuple, iterator, or string of CSV filename; the paths
        object should contain a series of lists or tuples (the paths)
    - Return: (generator)
    """
    if type(paths) != str:
        for path in paths:
            yield path
    if type(paths) == str:
        csv_file = open(paths)
        paths = csv.reader(csv_file, delimiter=',')
        for path in paths:
            path = [int(location) for location in path] 
            yield path
        csv_file.close()
        
def paths_all_or_set_dist(dist_matrix, paths, save=1, k="all"):
    """
    Calculate total distance for each supplied path.
    - Parameters:
      • dist_matrix: NumPy array; should be a symmetrix matrix like result
        from data_points_dist_mat()
      • paths: list, tuple, iterator, or string of CSV filename; the paths
        object should contain a series of lists or tuples (the paths)
      • save: 1 or string; default=1; if 1, save results to object; if string
        (filename), save results to CSV file (include the '.csv' extension)
      • k: "all" or integer for quantity of best/shortest paths to retain
    - Return:
      • if save=1: list of tuples; paths (save results to object)
      • if save=string: None (save results to CSV file)
    """
    # 4 possibilites: 1. save all to obj, 2. save k to obj, 3. save all to CSV, 4. save k to CSV
    n = dist_matrix.shape[0]
    paths = paths_all_or_set_strm(paths)
    data = []
    # save to object
    if save == 1:
        while True:
            try:
                path = next(paths)
                dist = [dist_matrix[(path[i],path[i+1])] for i in range(n-1)]
                dist = sum(dist)
                data.append((path,dist))
                data = sorted(data, key=lambda x: x[1])
                if type(k) == int and len(data) == k+1:
                    del data[-1]
            except:
                break
        return data
    # save to CSV file
    if type(save) == str:
        csv_file = open(save, 'w')
        csv_wrtr = csv.writer(csv_file, delimiter='\t')
        while True:
            try:
                path = next(paths)
                dist = [dist_matrix[(path[i],path[i+1])] for i in range(n-1)]
                dist = sum(dist)
                if k == "all":
                    csv_wrtr.writerow((path,dist))
                if type(k) == int:
                    data.append((path,dist))
                if type(k) == int and len(data) == k+1:
                    data = sorted(data, key=lambda x: x[1])
                    del data[-1]
            except:
                if type(k) == int:
                    for (path,dist) in data:
                        csv_wrtr.writerow((path,dist))
                csv_file.close()
                break
        return None
# =============================================================================


# =============================================================================
# 3 greedy approach: generate list of paths with distances
# -----------------------------------------------------------------------------
# def paths_list_greedy(file_save, dist_matrix, sort=True):
def paths_list_greedy(dist_matrix, save=1):
    """
    Using greedy approach and visit each data point once, generate list of
    paths and their total distances.
    - Unlike paths_list_all_or_set(), paths_list_greedy() gets paths and
      their distances together.
    - Greedy approach = always move to closest point not yet visited
      (aka 'nearest neighbor' [NN] method)
    - Quantity of paths = n.
    - Parameters:
      • dist_matrix: NumPy array; should be a symmetrix matrix like result
        from data_points_dist_mat()
      • save: 1 or string; default=1; if 1, save results to object; if string
        (filename), save results to CSV file (include the '.csv' extension)
    - Return:
      • if save=1: list of tuples; paths (save results to object)
      • if save=string: None (save results to CSV file)
    """
    n = dist_matrix.shape[0]
    dpIDs = tuple(range(n))
    paths = [{"path":None, "dist":None} for i in range(n)]
    # get paths and distances
    for dpID in dpIDs:
        # initialize path and dist
        path = [dpID]
        dist = np.float64(0) # or float(0)
        # setup for first step
        opts = list(dpIDs)
        opts.remove(dpID)
        data = list(zip(opts, dist_matrix[dpID][opts]))
        # take steps
        for i in range(n-1):
            step_path , step_dist = min(data, key=lambda x:x[1])
            path.append(step_path)
            dist += step_dist
            opts.remove(step_path)
            data = list(zip(opts, dist_matrix[step_path][opts]))
        paths[dpID]["path"] = path
        paths[dpID]["dist"] = dist
    paths = sorted(paths, key=lambda x: x["dist"])
    # save to object
    if save == 1:
        return paths
    # save to CSV file
    if type(save) == str:
        csv_file = open(save, 'w')
        csv_wrtr = csv.writer(csv_file, delimiter='\t')
        for x in paths:
            csv_wrtr.writerow((x["path"],x["dist"] ))
        csv_file.close()
        return None
# =============================================================================


# =============================================================================
# 4 plotting
# -----------------------------------------------------------------------------
def paths_plot(data_points, path_dist, save=None, cluster=None):
    """
    Plot the supplied points and path.
    - Parameters:
      • data_points: 2D NumPy array
      • path_dist: the path to plot; e.g., paths_list_greedy()[0]
      • save: string; save the plot as an image with the supplied filename
      • cluster: NumPy array; optional; cluster labels
    - Return: None
    """
    #### 0 setup
    n = data_points.shape[0]
    dpIDs = range(n)
    dp_x , dp_y = data_points[:,0] , data_points[:,1]
    path = path_dist["path"]
    dist = path_dist["dist"]
    matplotlib.rc('font', family='consolas')
    
    #### 1 title
    plt.title(s="Locations and Path \n (visit each point once)", weight='bold')

    #### 2 apply labels for data point IDs
    for label,x,y in zip(['' if x==path[0] else x for x in dpIDs],dp_x,dp_y):
        plt.annotate(
            s=label,
            color='k',
            xy=(x,y),
            xytext=(3,3),
            textcoords='offset pixels',
            fontsize=10,
        )
    
    #### 3 plot points - use this if cluster labels not provided
    if type(cluster) != np.ndarray:
        plt.scatter(
            dp_x,
            dp_y,
            s=10,
            c='k',
            marker='o',
        )

    #### 4 indicate clusters - use this if cluster labels provided
    if type(cluster) == np.ndarray:
        # 4.1 use distinct point colors and markers
        cluster_set = sorted(set(cluster))
        cluster_zippd = list(zip(range(n), cluster, dp_x, dp_y))
        cluster_split = {cluster:[] for cluster in cluster_set}
        for dp in cluster_zippd:
            for cluster in cluster_set:
                if dp[1] == cluster:
                    cluster_split[cluster].append(dp)
        cluster_split = [(k,v) for k,v in cluster_split.items()]
        cluster_split = sorted(cluster_split, key=lambda x: x[0])
        cluster_split = [b for a,b in cluster_split]
        clust_colrs = itertools.cycle("k b g m y c".split())
        clust_marks = itertools.cycle("o s ^ * D".split())
        p1 = plt.subplot()
        for cluster in cluster_split:
            next_colrs = next(clust_colrs)
            next_marks = next(clust_marks)
            p1.scatter(
                np.array([x for p,c,x,y in cluster]),
                np.array([y for p,c,x,y in cluster]),
                s=15,
                facecolor=next_colrs,
                edgecolor=next_colrs,
                marker=next_marks,
            )
        # 4.2 legend
        # legend_handls = broken
        legend_labels = [str(cluster)+" ("+str(size)+")" for cluster,size in 
            zip(cluster_set, [len(x) for x in cluster_split])]
        p1.legend(
            # handles=[],
            # labels=cluster_set,
            labels=legend_labels,
            title=str(len(cluster_set))+" clusters (n)",
            loc='upper left',
            bbox_to_anchor=(1, 1.02),
        )
    
    #### 5 indicate start point: label, marker, color
    # plt.annotate( # cancel arrow
    #     s='start',
    #     color='r',
    #     xy=(dp_x[path][0], dp_y[path][0]),
    #     xytext=(10,10),
    #     textcoords='offset pixels',
    #     fontsize=15,
    #     # arrowprops=dict(width=1, headwidth=1, headlength=1, facecolor='r', shrink=4, fc='r', ec='r'),
    # )
    plt.scatter(
        dp_x[path][0],
        dp_y[path][0],
        s=80,
        c='r',
        marker='x',
    )
    plt.annotate(
        s=path[0],
        color='r',
        xy=(dp_x[path][0], dp_y[path][0]),
        xytext=(-10,10),
        textcoords='offset pixels',
        fontsize=12,
    )
    
    #### 6 pathway arrows
    # 6.1 interpolation for arrows between points
    inter = np.array([[0,0]])
    threshold = (np.max(data_points) - np.min(data_points)) * 0.1
    for i in range(n-1):
        p_this , p_next = data_points[path][i] , data_points[path][i+1]
        inter = np.vstack((inter,p_this))
        distance = spatial.distance.pdist(X=(p_this,p_next))
        if distance > threshold:
            inter_n = int(distance/threshold) + 1
            if inter_n == 2:
                new = (p_this + p_next) / 2
            if inter_n != 2:
                new_x = np.linspace(start=p_this[0], stop=p_next[0],
                    num=inter_n, endpoint=False)
                new_y = np.linspace(start=p_this[1], stop=p_next[1],
                    num=inter_n, endpoint=False)
                new = np.column_stack((new_x,new_y))
                new = new[1:]
            inter = np.vstack((inter,new))
    inter = np.vstack((inter, data_points[path][-1])) # add last original point
    inter = inter[1:] # drop initial 0,0
    # 6.2 plot arrows
    for i in range(len(inter)-1):
        dp_x , dp_y = inter[:,0] , inter[:,1]
        this_x , this_y = dp_x[i]   , dp_y[i]
        next_x , next_y = dp_x[i+1] , dp_y[i+1]
        plt.annotate(
            s='',
            xy=(next_x, next_y),
            xytext=(this_x, this_y),
            arrowprops=dict(
                arrowstyle='-|>', fc='0.5', ec='0.5', ls=':', lw='1',),
        )
    
    #### 7 info text box
    info_text = "Info:" + '\n' \
        + "•N=" + str(n) + '\n' \
        + "•start:" + str(path[0]) + '\n' \
        + "•end  :" + str(path[-1]) + '\n' \
        + "•dist=" + '\n' \
        + " {:,.2f}".format(dist)
    plt.figtext(
        x=0.92, y=0.11,
        s=info_text,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(ec='k', fc='w'),
        fontdict=dict(size=16),
    )
    
    #### all done
    if save == None:
        plt.show()
    if type(save) == str:
        plt.savefig(filename=save, dpi=350, bbox_inches='tight',)
    return None
# =============================================================================

