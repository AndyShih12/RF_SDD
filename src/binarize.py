import os
import sys
import math
import glob
import pygraphviz as pygv
import sdd
from utils import timer
from utils import readConfig
from utils import util
import argparse
import collections

EPS = 1e-7

############################################################
# CLASSES
############################################################

class TreeState:
    def __init__(self,tree,filename):
        self.tree = tree
        self.filename = filename

        nodes = tree.nodes()
        root = tree.in_degree(nodes).index(0)
        root = nodes[root]
        self.root = root


def binarize_helper_node_list(node, tree):
    children = tree.successors(node)
    ret = []
    if len(children) == 0:
        pass
    else:
        for child in children:
            edge = tree.get_edge(node, child)
            var = node.attr['label'].split(' ')[-1]
            val = edge.attr['label'].split(' ')[-1]

            ret += [(var, (float)(val))]
            ret += binarize_helper_node_list(child, tree)
    return ret

def binarize(tree_states, header_test):
    pairs = []
    for tree_state in tree_states:
        pairs += binarize_helper_node_list(tree_state.root, tree_state.tree)
    pairs.sort() 
    pairs_dict = {}
    for k,v in pairs:
        if k not in pairs_dict.keys():
            pairs_dict[k] = [v]
        elif abs(v - pairs_dict[k][-1]) > EPS:
            pairs_dict[k] = pairs_dict[k] + [v]

    # discretize into coarse buckets
    pairs_dict_coarse = {}

    ''' 
    # fixed-bucket discretization
    for k,v in pairs_dict.iteritems():
        buckets = 4
        coarseness = len(v)/buckets if len(v) >= buckets else 1
        #print len(v), coarseness
        coarse_v = [x for i,x in enumerate(v) if i % coarseness == coarseness/2]
        pairs_dict_coarse[k] = coarse_v
    # end fixed-bucket discretization
    '''
    # epsilon-closeness discretization
    
    for k,v in pairs_dict.iteritems():
        ratio = threshold_ratio
        coarse_v = []
        for vv in v:
            if not coarse_v or (coarse_v[-1]*ratio < vv and vv*ratio > coarse_v[-1]):
                coarse_v.append(vv)
 
        pairs_dict_coarse[k] = coarse_v
    
    # end epsilon-closeness discretization

    pairs_dict = pairs_dict_coarse

    flip_header = [(v,k) for k,v in header_test.items()]
    sorted_header = [(k,v) for v,k in sorted(flip_header)[1:]]

    domain = collections.OrderedDict()
    sorted_pairs_dict = collections.OrderedDict()
    for k,_ in sorted_header:
        if k not in pairs_dict: pairs_dict[k] = [0] 
        v = pairs_dict[k]
        sorted_pairs_dict[k] = v
        for i in xrange(len(v)):
            domain[k + '_' + str(i)] = len(domain) + 1
        

    return domain, sorted_pairs_dict

def binarize_dataset(dataset, domain, pairs_dict, header, binarized_test_filename):
    binarized_dataset = []
    for data in dataset:
        binarized_data = [0 for _ in xrange(len(domain)+1)]
        for k,v in pairs_dict.iteritems():
            var = k
            for i,vv in enumerate(v):
                index = domain[var + "_" + str(i)]
                binarized_data[index] = (int)(data[header[var]] >= vv)

        binarized_data[0] = (int)(data[0] == 1)
        binarized_dataset.append(binarized_data)

    header_out = ["" for _ in xrange(len(domain)+1)]
    header_out[0] = "C"
    for k,v in domain.iteritems():
        header_out[v] = k
    
    filename = binarized_test_filename
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(",".join(header_out) + "\n")
        for d in binarized_dataset:
            f.write(",".join(str(x) for x in d) + "\n")

def discretize_dataset(dataset, domain, pairs_dict, header, discretized_test_filename):
    discretized_dataset = []
    header_out = ["C"]
    for k,v in pairs_dict.iteritems():
        header_out.append(k)

    for data in dataset:
        discretized_data = [0 for _ in xrange(len(pairs_dict)+1)]
        cnt = 1
        for k,v in pairs_dict.iteritems():
            var = k
            for i,vv in enumerate(v):
                if (data[header[var]] >= vv):
                  discretized_data[cnt] = i+1
            cnt += 1

        discretized_data[0] = (int)(data[0] == 1)
        discretized_dataset.append(discretized_data)

    filename = discretized_test_filename
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(",".join(header_out) + "\n")
        for d in discretized_dataset:
            f.write(",".join(str(x) for x in d) + "\n")


def binarize_tree_states(tree_states, domain, pairs_dict, binarized_tree_basename):
    for (k,tree_state) in enumerate(tree_states):
        tree = tree_state.tree
        for node in tree.nodes():
            if not len(tree.out_neighbors(node)):
                parts = node.attr['label'].split(" ")
                parts[2] = "1" if float(parts[2]) >= 0.5 else "0"
                node.attr['label'] = " ".join(parts)
                continue

            v = pairs_dict[node.attr['label'].split(' ')[-1]]
            base_label = node.attr['label']
            
            for j,edge in enumerate(tree.out_edges(node)):
                val = (float)(edge.attr['label'].split(' ')[-1])

                closest_i, closeness = 0, abs(val - v[0])
                for i,vv in enumerate(v):
                    if abs(val-vv) < closeness:
                        closeness = abs(val-vv)
                        closest_i = i
            
                new_label = " = %d" % (1 if edge.attr['label'].split(' ')[1] == ">=" else 0)
                edge.attr['label'] = new_label
                
                node.attr['label'] = base_label + "_" + str(closest_i)

        filename = binarized_tree_basename % k
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        tree.write(filename)

def write_constraints(domain, pairs_dict, constraint_filenames, constraint_sdd_filename, constraint_vtree_filename):
    constraints = []

    for k,v in pairs_dict.iteritems():
        cur = k
        constraints.append((cur,len(v)) + tuple(v))

    for filename in constraint_filenames:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename,"w") as f:
            f.write("num_variables: %d\n" % sum([c[1] for c in constraints]))
            for c in constraints:
                granularity = c[1]
                for i in xrange(granularity):
                    name = c[0]
                    thresh = c[i + 2]
                    f.write("name: %s_%d\n" % (name, i))
                    f.write("metadata: %s %f\n" % (name, thresh))

def run():
    with timer.Timer("reading dataset"):
        dataset_train = util.read_float_dataset(train_filename)
        header_train = util.read_header(train_filename)
        dataset_test = util.read_float_dataset(test_filename)
        header_test = util.read_header(test_filename)

    with timer.Timer("reading trees"):
        tree_states = []
        for filename in sorted(glob.glob(tree_basename.replace('%d', '*[0-9]'))):
            tree = pygv.AGraph(filename)
            tree_state = TreeState(tree, filename)
            tree_states.append(tree_state)
            #tree.layout(prog='dot')
            #tree.draw(filename+".png")
        #num_trees = len(tree_states)

    with timer.Timer("extracting binary variables"):
        domain, pairs_dict = binarize(tree_states, header_test)
        for tree_state in tree_states:
            tree_state.domain = domain
            tree_state.pairs_dict = pairs_dict
        binarize_dataset(dataset_train, domain, pairs_dict, header_train, binarized_train_filename)
        binarize_dataset(dataset_test, domain, pairs_dict, header_test, binarized_test_filename)
        discretize_dataset(dataset_train, domain, pairs_dict, header_train, discretized_train_filename)
        discretize_dataset(dataset_test, domain, pairs_dict, header_test, discretized_test_filename)


    with timer.Timer("binarizing trees"):
        binarize_tree_states(tree_states, domain, pairs_dict, binarized_tree_basename)

    with timer.Timer("writing constraints"):
        write_constraints(domain, pairs_dict, [constraint_filename_working, constraint_filename_output], constraint_sdd_filename, constraint_vtree_filename)

    print "\tdiscretization: "
    for k,v in pairs_dict.iteritems():
        print "\t",k,v
    #print domain


parser = argparse.ArgumentParser('Binarizes a continuous random forest into a discrete random forest.')
parser.add_argument('config_file', type=str, help='The config file.')
args = parser.parse_args()

if __name__ == '__main__':
    config_file = args.config_file
    config_json = readConfig.read_json(config_file)

    dataset = config_json["dataset"]
    tree_basename = config_json["rf_basename"]
    binarized_tree_basename = config_json["binarized_rf_basename"]
    train_filename = config_json["train_filename"]
    test_filename = config_json["test_filename"]
    binarized_train_filename = config_json["binarized_train_filename"]
    binarized_test_filename = config_json["binarized_test_filename"]
    discretized_train_filename = config_json["discretized_train_filename"]
    discretized_test_filename = config_json["discretized_test_filename"]
    constraint_filename_working = config_json["constraint_filename_working"]
    constraint_filename_output = config_json["constraint_filename_output"]
    constraint_sdd_filename = config_json["constraint_sdd_filename"]
    constraint_vtree_filename = config_json["constraint_vtree_filename"]
    num_trees = config_json["tree_count"]
    threshold_ratio = config_json["threshold_ratio"]

    run()      
