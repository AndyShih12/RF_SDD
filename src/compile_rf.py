#!/usr/bin/env python

import sys
import math
import glob
import pygraphviz as pygv
import sdd
from utils import util
from utils import readConfig
from utils import timer
from itertools import izip
import argparse

#EPS = 1e-7

############################################################
# CLASSES
############################################################

class TreeState:
    def __init__(self,tree,domain=None,constraint_info=None,constraint_sdd=None):
        self.tree = tree
        self.domain = domain
        self.constraint_info = constraint_info
        self.constraint_sdd = constraint_sdd

        nodes = tree.nodes()
        root = tree.in_degree(nodes).index(0)
        root = nodes[root]
        self.root = root

class SddState:
    def __init__(self,vtree,manager):
        self.vtree = vtree
        self.manager = manager
        self.alpha = sdd.sdd_manager_false(manager)
        self.used_vars = set()

class SddSizeStats:
    def __init__(self):
        self.count = 0
        self.min_size = 1e400 #inf
        self.max_size = 0
        self.sum_size = 0

    def __repr__(self):
        st = []
        st.append("   count: %d" % self.count)
        st.append("min size: %f" % self.min_size)
        st.append("max size: %d" % self.max_size)
        st.append("avg size: %.2f" % (float(self.sum_size)/(1 if (self.count == 0) else self.count)))
        return "\n".join(st)

    def update(self,alpha):
        size = sdd.sdd_size(alpha)
        self.count += 1
        if size < self.min_size: self.min_size = size
        if size > self.max_size: self.max_size = size
        self.sum_size += size

############################################################
# COMPILE DECISION TREES
############################################################

def compile_tree(node,tree_state,sdd_state,label="0",st=None,path_sdd=None):
    if st is None: st = ""
    mgr = sdd_state.manager
    if path_sdd is None:
        path_sdd = sdd.sdd_manager_true(mgr)

    tree = tree_state.tree
    children = tree.successors(node)
    if len(children) == 0:
        node_label = node.attr['label']
        node_label = node_label.split(':')[-1].strip().split(' ')[0]
        if label == node_label:
            # disjoin path
            #print st
            alpha = sdd.sdd_disjoin(sdd_state.alpha,path_sdd,mgr)
            sdd.sdd_deref(sdd_state.alpha,mgr)
            sdd.sdd_ref(alpha,mgr)
            sdd_state.alpha = alpha
    else:
        for child in children:
            edge = tree.get_edge(node,child)
            var = node.attr['label'].split(' ')[-1]
            val = edge.attr['label'].split(' ')[-1]
            child_st = st + "%s:%s " % (var,val)

            # extend path
            base_var = "_".join(var.split('_')[:-1]) + "_%d"
            cur_index = int(var.split('_')[-1])
            low_index, high_index = 0, tree_state.constraint_info[base_var][0]

            beta = sdd.sdd_manager_false(mgr)
            if val == ">=":
                for i in xrange(cur_index + 1, high_index):
                    sdd_lit = tree_state.domain[base_var % i]
                    beta = sdd.sdd_disjoin(beta, sdd.sdd_manager_literal(sdd_lit,mgr), mgr)
            else: # val == "<"
                for i in xrange(low_index, cur_index + 1):
                    sdd_lit = tree_state.domain[base_var % i]
                    beta = sdd.sdd_disjoin(beta, sdd.sdd_manager_literal(sdd_lit,mgr), mgr)

            constraint_sdd = tree_state.constraint_sdd

            sdd_var = tree_state.domain[var]
            new_path_sdd = sdd.sdd_conjoin(path_sdd,beta,mgr)
            sdd_state.used_vars.add(sdd_var)

            child_st = st + "%s:%s " % (var, val)
            sdd.sdd_ref(new_path_sdd,mgr)
            compile_tree(child,tree_state,sdd_state,
                         label=label,st=child_st,path_sdd=new_path_sdd)
            sdd.sdd_deref(new_path_sdd,mgr)

############################################################
# COMPILE ALL
############################################################

def forest_sdds_iter(tree_states,sdd_state):
    false_sdd = sdd.sdd_manager_false(sdd_state.manager)
    for i,tree_state in enumerate(tree_states):
        sdd_state.alpha = false_sdd
        sdd_state.used_vars = set()
        compile_tree(tree_state.root,tree_state,sdd_state,label="1")
        #print sdd.sdd_global_model_count(sdd_state.alpha, sdd_state.manager)
        ''' 
        if OPTIONS.majority_circuit_opt:
            mgr = sdd_state.manager
            domain = tree_state.domain
            var = sdd.sdd_manager_literal(domain["Tree_%d" % i], mgr)
            sdd.sdd_ref(var, mgr)            

            alpha = sdd.sdd_conjoin(
                sdd.sdd_disjoin(sdd_state.alpha, sdd.sdd_negate(var, mgr), mgr),
                sdd.sdd_disjoin(sdd.sdd_negate(sdd_state.alpha, mgr), var, mgr),
                mgr
            )
            sdd.sdd_deref(var, mgr)
            sdd.sdd_deref(sdd_state.alpha, mgr)
            sdd.sdd_ref(alpha, mgr)
            sdd_state.alpha = alpha
        '''

        yield sdd_state.alpha, sdd_state.used_vars

def encode_unique_constraint(values, mgr):
    alpha = sdd.sdd_manager_true(mgr)

    # at most one
    for v1 in values:
        for v2 in values:
            if v1 == v2: continue
            beta = sdd.sdd_disjoin(sdd.sdd_manager_literal(-1*v1,mgr), sdd.sdd_manager_literal(-1*v2,mgr), mgr)
            alpha = sdd.sdd_conjoin(alpha, beta, mgr)

    # at least one
    beta = sdd.sdd_manager_false(mgr)
    for v in values:
      beta = sdd.sdd_disjoin(beta, sdd.sdd_manager_literal(v,mgr), mgr) 
    alpha = sdd.sdd_conjoin(alpha, beta, mgr)

    return alpha

def encode_logical_constraints(constraint_filename, mgr, domain):
    with open(constraint_filename, "r") as f:
        lines = f.readlines()

    lines = [line.strip().split(" ")[1:] for line in lines]
    num_variables = int(lines[0][0])

    constraints = {}
    cur = None

    for i in xrange(num_variables):
        name = lines[2*i+1][0]
        metadata = lines[2*i+2]
        key, thresh = metadata[0], float(metadata[1])

        if cur and cur[0] != key:
            constraints[cur[0] + "_%d"] =  (len(cur[1]),) + tuple(cur[1]) 
            cur = None

        if not cur:
            cur = (key, [thresh])
        else:
            cur[1].append(thresh)
    constraints[cur[0] + "_%d"] =  (len(cur[1]),) + tuple(cur[1])
 
    #print constraints

    alpha = sdd.sdd_manager_true(mgr)

    for k,v in constraints.items():
        high = v[0]
        values = [domain[k % i] for i in xrange(0, high)]

        beta = encode_unique_constraint(values, mgr)
        alpha = sdd.sdd_conjoin(alpha, beta, mgr)

    return alpha, constraints


def pick_next_tree(used_vars_list, used_vars):
    min_vars_cnt, next_tree_index = len(used_vars_list[0]), 0

    # minimize use of new variables heuristic
    for i, cur_set in enumerate(used_vars_list):
        new_vars_cnt = len(cur_set - used_vars)
        if new_vars_cnt < min_vars_cnt:
              min_vars_cnt, next_tree_index = new_vars_cnt, i

    return next_tree_index

def compile_all(forest_sdds,used_vars_list,num_trees,domain,manager,constraint_sdd=None):
    half = int(math.ceil(num_trees/2.0))
    true_sdd = sdd.sdd_manager_true(manager)
    false_sdd = sdd.sdd_manager_false(manager)
    last_size = 2**16

    if not constraint_sdd:
      constraint_sdd = sdd.sdd_manager_true(manager)
    true_sdd = constraint_sdd

    sdd.sdd_ref(true_sdd,manager)

    to_compile_sdds = [tree_sdd for tree_sdd in forest_sdds]
    used_vars_list = [used_vars for used_vars in used_vars_list]

    '''
    if OPTIONS.majority_circuit_opt:
        majority_sdds = [sdd.sdd_manager_literal(domain["Tree_%d" % i], manager) for i in xrange(num_trees)]    
        for single_sdd in majority_sdds:
            sdd.sdd_ref(single_sdd, manager)
        to_compile_sdds = majority_sdds
        used_vars_list = [set() for _ in forest_sdds]
    '''

    cur = [ true_sdd, false_sdd ]
    used_vars = set()

    for k in xrange(num_trees):
        last,cur = cur,[]

        tree_index = pick_next_tree(used_vars_list, used_vars)
        tree_sdd = to_compile_sdds[tree_index]
        used_vars |= used_vars_list[tree_index]
        to_compile_sdds = to_compile_sdds[:tree_index] + to_compile_sdds[tree_index+1:]
        used_vars_list = used_vars_list[:tree_index] + used_vars_list[tree_index+1:]
        
        for i in xrange(min(half,k+1)+1):
            cur_sdd = last[i]
            #cur_sdd = sdd.sdd_conjoin(sdd.sdd_negate(tree_sdd,manager),cur_sdd,manager)
            """
            elif i+(num_trees-k) < half: # don't bother
                cur_sdd = sdd.sdd_manager_false(manager)
            """
            if i == 0:
                pass
            elif i > 0:
                alpha = sdd.sdd_conjoin(tree_sdd,last[i-1],manager)
                sdd.sdd_deref(last[i-1],manager)
                cur_sdd = sdd.sdd_disjoin(cur_sdd,alpha,manager)
            sdd.sdd_ref(cur_sdd,manager)
            cur.append(cur_sdd)

            if sdd.sdd_manager_dead_count(manager) >= 2*last_size:
                sdd.sdd_manager_garbage_collect(manager)
            if sdd.sdd_manager_live_count(manager) >= 2*last_size:
                print "*",
                sdd.sdd_manager_minimize_limited(manager)
                last_size = 2*last_size


        if k >= half: sdd.sdd_deref(last[-2],manager)
        sdd.sdd_deref(tree_sdd,manager)
        cur.append(false_sdd)

        print "%d" % (num_trees-k),
        sys.stdout.flush()
        #print "%d/%d" % (k,num_trees)
        print "live size:", sdd.sdd_manager_live_count(manager)
        #print "dead size:", sdd.sdd_manager_dead_count(manager)
        sdd.sdd_manager_garbage_collect(manager)
        #sdd.sdd_manager_minimize_limited(manager)

    #for alpha in cur: sdd.sdd_deref(alpha,manager)
    ret = cur[-2]

    '''
    if OPTIONS.majority_circuit_opt:
        # save ret (the majority circuit)
        # save each individual tree_sdd
        vtree = sdd.sdd_manager_vtree(manager)
        majority_sdd_filename = "%s_majority.sdd" % sdd_basename
        majority_vtree_filename = "%s_majority.vtree" % sdd_basename
        print "Writing majority sdd file %s and majority vtree file %s" % (majority_sdd_filename, majority_vtree_filename)
        sdd.sdd_save(majority_sdd_filename,ret)
        sdd.sdd_vtree_save(majority_vtree_filename,vtree)

        print "Writing individual tree sdds..."
        for k,tree_sdd in enumerate(forest_sdds):
            tree_name = "tree_%d" % k
            tree_sdd_filename = "%s_majority_%s.sdd" % (sdd_basename, tree_name)
            sdd.sdd_save(tree_sdd_filename, tree_sdd)

        gamma = sdd.sdd_manager_true(manager)
        for k,tree_sdd in enumerate(forest_sdds):
            new_gamma = sdd.sdd_conjoin(gamma, tree_sdd, manager)
            sdd.sdd_ref(new_gamma, manager)
            sdd.sdd_deref(gamma, manager)
            gamma = new_gamma

            if sdd.sdd_manager_dead_count(manager) >= 2*last_size:
                sdd.sdd_manager_garbage_collect(manager)
            if sdd.sdd_manager_live_count(manager) >= 2*last_size:
                print "*",
                sdd.sdd_manager_minimize_limited(manager)
                last_size = 2*last_size

            print "%d" % k,
            sys.stdout.flush()
            print "live size:", sdd.sdd_manager_live_count(manager)
        ret = sdd.sdd_conjoin(ret, gamma, manager)
        
        #existential quantification
        print "Existential quantification..."
        exists_map = sdd.new_intArray(len(domain))
        for i in xrange(len(domain)):
            sdd.intArray_setitem(exists_map,i,0)
        for i in xrange(num_trees):
            lit = domain["Tree_%d" % i]
            sdd.intArray_setitem(exists_map,lit,1)
        ret = sdd.sdd_exists_multiple(exists_map, ret, manager)
    '''

    return ret

############################################################
# CHECK MONOTONICITY
############################################################

def is_monotone(alpha,manager):
    """ A function alpha is monotone iff f|~x |= f|x for all X.
    """
    result = True
    var_count = sdd.sdd_manager_var_count(manager)
    sdd.sdd_ref(alpha,manager)
    for x in xrange(1,var_count+1):
        f_x  = sdd.sdd_condition(x,alpha,manager)
        sdd.sdd_ref(f_x,manager)
        f_nx = sdd.sdd_condition(-x,alpha,manager)
        sdd.sdd_ref(f_nx,manager)
        # f|~x |= f|x iff f|~x ^ f|x == f|~x
        f_both = sdd.sdd_conjoin(f_x,f_nx,manager)
        if f_both != f_x:
            result = False
        sdd.sdd_deref(f_x,manager),sdd.sdd_deref(f_nx,manager)
        if result is False: break
    sdd.sdd_deref(alpha,manager)
    return result

############################################################
# MAIN
############################################################

def run():
    with timer.Timer("reading dataset"):
        dataset = util.read_binary_dataset(test_filename)
        domain = util.read_header(test_filename)

        '''
        if OPTIONS.majority_circuit_opt:
            l = len(domain)
            for k in xrange(num_trees):
                domain["Tree_%d" % k] = l+k
        '''

    with timer.Timer("initializing manager"):
        # start sdd manager
        var_count = len(domain) - 1
        vtree = sdd.sdd_vtree_new(var_count,"balanced")
        manager = sdd.sdd_manager_new(vtree)
        #sdd.sdd_manager_auto_gc_and_minimize_on(manager)
        #sdd.sdd_manager_auto_gc_and_minimize_off(manager)
        sdd_state = SddState(vtree,manager)

    with timer.Timer("reading constraints"):
        constraint_sdd, constraint_info = encode_logical_constraints(constraint_filename,manager,domain)
        sdd.sdd_ref(constraint_sdd,manager)

    with timer.Timer("reading trees"):
        tree_states = []
        for filename in sorted(glob.glob(tree_basename.replace('%d','*'))):
            tree = pygv.AGraph(filename)
            tree_state = TreeState(tree,domain,constraint_info,constraint_sdd)
            tree_states.append(tree_state)
            #tree.layout(prog='dot')
            #tree.draw(filename+".png")
        #num_trees = len(tree_states)

    with timer.Timer("compiling trees"):
        forest_sdds, _ = izip(*forest_sdds_iter(tree_states,sdd_state))
        #forest_sdds = list(forest_sdds_iter(tree_states,sdd_state))

        forest_sdds = [ (tree_state,tree_sdd) for tree_state,tree_sdd in
                        zip(tree_states,forest_sdds) ]
        cmpf = lambda x,y: cmp(sdd.sdd_size(x[1]),sdd.sdd_size(y[1]))
        forest_sdds.sort(cmp=cmpf)
        tree_states = [ tree_state for tree_state,tree_sdd in forest_sdds ]

        #ACACAC
        sdd.sdd_manager_auto_gc_and_minimize_off(manager)
        sdd.sdd_manager_minimize_limited(manager)
        stats = SddSizeStats()
        for tree_state,tree_sdd in forest_sdds:
            stats.update(tree_sdd)
            sdd.sdd_deref(tree_sdd,manager)
        sdd.sdd_manager_garbage_collect(manager)
        forest_sdds, used_vars_list = izip(*forest_sdds_iter(tree_states,sdd_state))
    print stats

    with timer.Timer("compiling all",prefix="| "):
        alpha = compile_all(forest_sdds,used_vars_list,num_trees,domain,manager,constraint_sdd)

    with timer.Timer("evaluating"):
        msg = util.evaluate_dataset_all_sdd(dataset,alpha,manager)
    print "|     trees : %d" % num_trees
    print "--- evaluating majority vote on random forest (compiled):"
    print msg
    print "|  all size :", sdd.sdd_size(alpha)
    print "|  all count:", sdd.sdd_count(alpha)
    print " model count:", sdd.sdd_global_model_count(alpha,manager)

    with timer.Timer("checking monotonicity"):
        result = is_monotone(alpha,manager)
    print "Is monotone?", result

    #for tree_sdd in forest_sdds: sdd.sdd_deref(tree_sdd,manager)
    print "===================="
    print "before garbage collecting..." 
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)
    print "garbage collecting..."
    sdd.sdd_manager_garbage_collect(manager)
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)

    vtree = sdd.sdd_manager_vtree(manager)
    print "Writing sdd file %s and vtree file %s" % (sdd_filename, vtree_filename)
    sdd.sdd_save(sdd_filename,alpha)
    sdd.sdd_vtree_save(vtree_filename,vtree)

    print "Writing constraint sdd file %s and constraint vtree file %s" % (constraint_sdd_filename, constraint_vtree_filename)
    sdd.sdd_save(constraint_sdd_filename,constraint_sdd)
    sdd.sdd_vtree_save(constraint_vtree_filename,vtree)


parser = argparse.ArgumentParser('Compiles a random forest into an SDD.')

parser.add_argument('config_file', type=str, help='The config file.')
args = parser.parse_args()

if __name__ == '__main__':
    config_file = args.config_file
    config_json = readConfig.read_json(config_file)

    tree_basename = config_json["binarized_rf_basename"]
    test_filename = config_json["binarized_test_filename"]
    constraint_filename = config_json["constraint_filename_working"]
    sdd_filename = config_json["sdd_filename"]
    vtree_filename = config_json["vtree_filename"]
    constraint_sdd_filename = config_json["constraint_sdd_filename"]
    constraint_vtree_filename = config_json["constraint_vtree_filename"]
    num_trees = config_json["tree_count"]

    run()
