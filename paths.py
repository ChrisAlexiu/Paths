"""
Pathing

Given a set of data points (2D), identify paths to visit each point once and determine the distance traveled. Related: traveling salesman problem.
"""

import itertools
import csv
import time
import datetime
import multiprocessing
import os
import numpy as np
from scipy import spatial
# import matplotlib
# import matplotlib.pyplot as plt

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
        csv_file_w = open(save, 'w')
        csv_writer = csv.writer(csv_file_w, delimiter=',')
        for path in paths:
            csv_writer.writerow((path))
        csv_file_w.close()
        return None
# =============================================================================


# =============================================================================
# 2. utility function for calculating path distance 
# -----------------------------------------------------------------------------
    """
    Utility function for calculating path distance.
    - Parameters:
      • path: list or tuple
    - Return:
      • dist: distance; the distance for the given path calculated from the given
        distance matrix
    """

    # experimental, 1
    dist = np.sum(dist)
    return dist
# =============================================================================


# =============================================================================
# 3. calculate distance for 'all' or 'set' paths
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
        csv_file_r = open(paths)
        paths = csv.reader(csv_file_r, delimiter=',')
        for path in paths:
            path = tuple([int(location) for location in path])
            yield path
        csv_file_r.close()
    return None

        
    """
    Calculate total distance for each supplied path.
    - Parameters:
      • paths: list, tuple, iterator, or string of CSV filename; the paths
        object should contain a series of lists or tuples (the paths)
      • save: 1 or string; default=1; if 1, save results to object; if string
        (filename), save results to CSV file (include the '.csv' extension)
      • k: "all" or integer for quantity of best/shortest paths to retain
    - Return: None - generator or CSV file of paths and distances
    """
    # 4 possibilites:
    # 1. save all to obj, 2. save k to obj, 3. save all to CSV, 4. save k to CSV
    paths = paths_all_or_set_strm(paths)
    kdata = [] # used only for best k
    # save to object
    if save == 1:
    # save to CSV file
    if type(save) == str:
    return None


    """
    Helper function for paths_all_or_set_dist(), used to save results to object.
    """
    for path in paths:
        if k == "all":
            yield (path,dist)
        if type(k) == int:
            kdata.append((path,dist))
        if type(k) == int and len(kdata) == k+1:
             kdata = sorted(kdata, key=lambda x: x[1])
             del kdata[-1]
    if type(k) == int:
        for path_dist in kdata:
            yield path_dist
    return None


    """
    Helper function for paths_all_or_set_dist(), used to save results to CSV file.
    """
    csv_file_w = open(save, 'w')
    csv_writer = csv.writer(csv_file_w, delimiter='\t')
    for path in paths:
        if k == "all":
            csv_writer.writerow((path,dist))
        if type(k) == int:
            kdata.append((path,dist))
        if type(k) == int and len(kdata) == k+1:
             kdata = sorted(kdata, key=lambda x: x[1])
             del kdata[-1]
    if type(k) == int:
        for path_dist in kdata:
            csv_writer.writerow((path_dist))
    return None
# =============================================================================


# =============================================================================
# 4 greedy approach: generate list of paths with distances
# -----------------------------------------------------------------------------
    """
    Using greedy approach and visit each data point once, generate list of
    paths and their total distances.
    - Unlike paths_list_all_or_set(), paths_list_greedy() gets paths and
      their distances together.
    - Greedy approach = always move to closest point not yet visited
      (aka 'nearest neighbor' [NN] method)
    - Quantity of paths = n.
    - Parameters:
      • save: 1 or string; default=1; if 1, save results to object; if string
        (filename), save results to CSV file (include the '.csv' extension)
    - Return:
      • if save=1: list of tuples; paths (save results to object)
      • if save=string: None (save results to CSV file)
    """
    paths = [None] * n
    # get paths and distances
    for i in range(n):
        # initialize path and setup for first step
        path = [i]
        opts = list(range(n))
        opts.remove(i)
        # take steps
        for j in range(n-1):
            step = min(data, key=lambda x: x[1])[0]
            path.append(step)
            opts.remove(step)
        paths[i] = path
    paths = sorted(paths, key=lambda x: x[1])
    # save to object
    if save == 1:
        return paths
    # save to CSV file
    if type(save) == str:
        csv_file_w = open(save, 'w')
        csv_wrtr = csv.writer(csv_file_w, delimiter='\t')
        for x in paths:
            csv_wrtr.writerow((x[0],x[1]))
        csv_file_w.close()
        return None
# =============================================================================


# =============================================================================
# 5 improve an identified path via point swapping
# -----------------------------------------------------------------------------
    max_iter=None, max_time=None, parallel=False, report=False):
    """
    Try to find an improved path from a reference path by swapping pairs of 
    sequences of one or more locations.
    - Given a path, swap a pair of single or multiple locations, then calculate 
      the distance. Do this for all possible swaps. Choose the best swap. Iterate 
      until max_iter, max_time, or no better paths can be found.
    - Parameters:
      • path_dist: 2-tuple; 1. path, 2. distance
      • output: string; "hist" (history) or "best"; default="hist"
      • max_iter: None or integer; default=None
      • max_time: None or string; default=None; if string, use input like "30s",
        "2m", or "1h" for however many seconds, minutes, or hours, respectively
      • parallel: boolean or integer; default=None; set this to use 
        multiprocessing and the number of processes to use
      • report: boolean; default=False; if True, when finished, print a brief 
        report containing runtime and the distance delta
    - Return:
      • if output="hist: list containing all step-wise improvements
      • if output="best: 2-tuple containing best path and distance
    """
    t1 = time.time()
    path_init , dist_init = path_dist
    n = len(path_init)
        
    #### changes type 1 - pairs of single-location swaps
    ct1 = itertools.product(range(n), repeat=2)
    ct1 = [(a,b) for a,b in ct1 if a<b]
    ct1 = [((a,a+1),(b,b+1),1,1) for a,b in ct1]
        
    #### changes type 2 - pairs of multi-location swaps with all 4 orderings
    ct2 = []
    for seq_len in range(2,int(n/2)+1):
        # get all index k-seqs
        temp = [list(range(n))[i:i+seq_len] for i in range(n+1-seq_len)]
        # get all index k-seqs termini (inclusive,exclusive)
        temp = [(x[0],x[-1]+1) for x in temp]
        # get cartesian product for all index k-seqs termini
        temp = list(itertools.product(temp, repeat=2))
        # filter to remove self-swaps, overlapping-swaps, and reverse-swaps
            # e.g., remove: (0,0),(0,0); (0,2),(1,3); (0,2),(2,0)
        temp = [(a,b) for a,b in temp
            if a!=b and not a[0]<b[0]<a[1] and not b[0]<a[0]<b[1]]
        temp = [tuple(sorted(x)) for x in temp]
        temp = sorted(set(temp))
        # get cartesian product of each index k-seq with all 4 orderings
        temp = itertools.product(temp,itertools.product([1,-1],repeat=2))
        # reduce nesting
        temp = [(a,b,x,y) for (a,b),(x,y) in temp]
        # done
        ct2.append(temp)
    ct2 = list(itertools.chain.from_iterable(ct2))
    
    # changes type 3 - multi-location inversions
    ct3 = []
    for seq_len in range(2,n):
        temp = [list(range(n))[i:i+seq_len] for i in range(n+1-seq_len)]
        temp = [(x[0],x[-1]+1) for x in temp]
        temp = [((a,b),(a,b),-1,-1) for a,b in temp]
        ct3.append(temp)
    ct3 = list(itertools.chain.from_iterable(ct3))
    
    #### final prep
    ct_all = list(itertools.chain(ct1, ct2, ct3))
    n_options = len(ct_all)
    del ct1, ct2, ct3
    
    n_cpu = os.cpu_count()
    if parallel in (True, "True", "true", "T", "t"):
        parallel , n_processes = True , n_cpu-1
    if type(parallel) == int and parallel != 0:
        parallel , n_processes = True , parallel if parallel<=n_cpu else n_cpu
    if parallel in (False, "False", "false", "F", "f", 0):
        parallel , n_processes = False , "N/A"
    del n_cpu
    
    max_time_report = max_time
    if max_time != None:
        te_denominator = dict(zip("s m h".split() , (1, 60, 60**2)))
        te_denominator = te_denominator[max_time[-1]]
        max_time = float(max_time[:-1])

    path_best , dist_best = list(path_init) , dist_init
    history = []
    history.append((path_best,dist_best,*["N/A"]*4))
    loop_i = -1

    ##### do it
    while True:
        loop_i += 1
        if max_iter != None and loop_i == max_iter:
            exit_cond = "max_iter"
            break
        if max_time != None: 
            t2 = (time.time() - t1) / te_denominator
            if t2 > max_time:
                exit_cond = "max_time"
                break
        
        iterthis = list(zip(
            itertools.repeat(path_best, n_options),
            itertools.repeat(dist_best, n_options),
            ct_all,
        ))

        if parallel == True:
            mp_pool = multiprocessing.Pool(processes=n_processes)
            history_iter = mp_pool.map(paths_change_improve_main, iterthis)
            mp_pool.close()
            mp_pool.join()
        
        if parallel == False:
            history_iter = [paths_change_improve_main(x) for x in iterthis]
        
        history_iter = [x for x in history_iter if x!=None]
        
        # pick best from each iter - use best as start point for next iter
        if len(history_iter) == 0:
            exit_cond = "finished"
            break
        this_iter_best = min(history_iter, key=lambda x:x[1])
        history.append(this_iter_best)
        path_best , dist_best = this_iter_best[0:2]
    
    # reporting
    if report in (True, "True", "true", "T", "t", 1):
        t2 = str(datetime.timedelta(seconds=time.time()-t1))
        t2 = t2[:t2.index(".")]
        print("Report")
        print("Exit b/c:", exit_cond)
        print("Runtime :", t2)
        print("Max_Time:", max_time_report)
        print("Max_Iter:", max_iter)
        print("Parallel:", parallel, "• (n processes = {:})".format(n_processes))
        print("Options :", "{:,}".format(n_options))
        print("Path Len:", "{:,}".format(n))
        if path_best != list(path_init):
            print("Result  : A better/shorter path was found", end="") 
            print(" - {:} change(s) were made:".format(len(history)-1))
            print("  • Initial: {:10,.2f} unit".format(dist_init))
            print("  • Final  : {:10,.2f} unit".format(dist_best))
            print("  • Delta x: {:10,.2f} unit".format(dist_init-dist_best))
            print("  • Delta %: {:10,.2f} %"\
                .format((dist_init-dist_best)/dist_init*100))
        if path_best == list(path_init):
            print("A better/shorter path could not be found.")
    
    # done
    if output == "hist":
        return history
    if output == "best":
        return (history[-1][0],history[-1][1])


def paths_change_improve_main(x_from_iterthis):
    """
    For modularity and to enable multiprocessing, main action of
    paths_change_improve() was put into this function.
    """
    path_next = list(path_best)

    a , b , x , y = swap
    path_next[slice(*a)] , path_next[slice(*b)] = \
        path_next[slice(*b)][::x] , path_next[slice(*a)][::y]
    
    if path_next == path_best[::-1]:
        return None
    
    if dist_next < dist_best:
        return (path_next,dist_next,a,b,x,y)
    else:
        return None
# =============================================================================


# =============================================================================
# 6 plotting
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
    
    import matplotlib
    import matplotlib.pyplot as plt
    
    #### 0 setup
    n = data_points.shape[0]
    dpIDs = range(n)
    dp_x , dp_y = data_points[:,0] , data_points[:,1]
    # path = path_dist["path"]
    # dist = path_dist["dist"]
    path , dist = path_dist
    matplotlib.rc('font', family='consolas')
    
    #### 1 title
    plt.title(s="Locations and Path (visit each point once)", weight='bold')

    #### 2 apply labels for data point IDs
    for label, x, y in \
        zip(['' if x==path[0] or x==path[-1] else x for x in dpIDs], dp_x, dp_y):
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
            bbox_to_anchor=(1.19, 1.02),
        )
    
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
        dp_x[path[0]],
        dp_y[path[0]],
        s=80,
        c='r',
        marker='x',
    )
    for i in [0, -1]:
        plt.annotate(
            s=path[i],
            color='r',
            xy=(dp_x[path[i]], dp_y[path[i]]),
            xytext=(-10,10),
            textcoords='offset pixels',
            fontsize=12,
            weight='bold'
        )
    
    #### 6 pathway arrows
    # 6.1 interpolation for arrows between points
    inter = np.array([[0,0]])
    threshold = (np.max(data_points) - np.min(data_points)) * 0.1
    for i in range(n-1):
        p_this , p_next = data_points[path[i]] , data_points[path[i+1]]
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
    inter = np.vstack((inter, data_points[path[-1]])) # add last original point
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
        # x=0.92, y=0.11,
        x=0.92, y=0.89,
        s=info_text,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(ec='k', fc='w'),
        fontdict=dict(size=14),
    )
    
    #### all done
    if save == None:
        plt.show()
    if type(save) == str:
        plt.savefig(filename=save, dpi=350, bbox_inches='tight',)
    plt.clf()
    return None
# =============================================================================

