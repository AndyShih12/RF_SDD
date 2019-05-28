import sdd

############################################################
# SDD EVALUATION
############################################################

def sdd_node_elements(node):
    size = 2*sdd.sdd_node_size(node)
    elements = sdd.sdd_node_elements(node)
    return [ sdd.sddNodeArray_getitem(elements,i) for i in xrange(size) ]

def node_value(node,values,model):
    if sdd.sdd_node_is_true(node):
        return True
    elif sdd.sdd_node_is_false(node):
        return False
    elif sdd.sdd_node_is_literal(node):
        lit = sdd.sdd_node_literal(node)
        var = lit if lit > 0 else -lit
        lit = 1 if lit > 0 else 0
        return model[var] == lit
    else:
        sdd_id = sdd.sdd_id(node)
        return values[sdd_id]

def test_model(model,alpha,manager,values=None):
    if values is None: values = {}
    sdd_id = sdd.sdd_id(alpha)
    if sdd_id in values:
        return values[sdd_id]
    if not sdd.sdd_node_is_decision(alpha): # is terminal
        value = node_value(alpha,values,model)
    else: # is decomposition
        value = False
        elements = sdd_node_elements(alpha)
        for p,s in pairs(elements):
            pval = test_model(model,p,manager,values=values)
            if pval is False: continue
            sval = test_model(model,s,manager,values=values)
            if sval is True: value = True
            break
    values[sdd_id] = value
    return value

############################################################
# TEST
############################################################

def evaluate_instance(instance,forest_sdds,manager):
    pos_count,neg_count = 0,0
    for tree_sdd in forest_sdds:
        label = test_model(instance,tree_sdd,manager)
        if label: pos_count += 1
        else: neg_count += 1
    return pos_count >= neg_count # ACACAC

def evaluate_dataset(dataset,forest_sdds,manager):
    correct,incorrect = 0,0
    for instance in dataset:
        true_label = instance[0] == 1
        pred_label = evaluate_instance(instance,forest_sdds,manager)
        if true_label == pred_label: correct += 1
        else: incorrect += 1
    st = "  correct : %d\nincorrect : %d" % (correct,incorrect)
    return st

def evaluate_dataset_all_sdd(dataset,alpha,manager):
    correct,incorrect = 0,0
    for instance in dataset:
        true_label = instance[0] == 1
        pred_label = test_model(instance,alpha,manager)
        #print instance
        #print true_label, pred_label
        if true_label == pred_label: correct += 1
        else: incorrect += 1
    st = "  correct : %d\nincorrect : %d" % (correct,incorrect)
    return st

############################################################
# UTIL
############################################################

def pairs(my_list):
    """a generator for (prime,sub) pairs"""
    if my_list is None: return
    it = iter(my_list)
    for x in it:
        y = it.next()
        yield (x,y)

def read_header(filename):
    with open(filename,'r') as f:
        header = f.readline().strip().split(',')
        domain = {}
        for i,name in enumerate(header):
            domain[name] = i
    return domain

def read_float_dataset(filename):
    models = []
    with open(filename,'r') as f:
        f.readline() # header
        for line in f.readlines():
            line = line.strip().split(',')
            line = [ x for x in line ]
            model = { i:float(x) for i,x in enumerate(line) }
            models.append(model)
    return models

def read_binary_dataset(filename):
    models = []
    with open(filename,'r') as f:
        f.readline() # header
        for line in f.readlines():
            line = line.strip().split(',')
            line = [ x for x in line ]
            model = { i:int(x) for i,x in enumerate(line) }
            models.append(model)
    return models

class CompileOptions(object):
    def __init__(self, **kwargs):
        if "logarithmic_opt" in kwargs:
            self.logarithmic_opt = kwargs["logarithmic_opt"]
        if "majority_circuit_opt" in kwargs:
            self.majority_circuit_opt = kwargs["majority_circuit_opt"]
        if "threshold_ratio" in kwargs:
            self.threshold_ratio = kwargs["threshold_ratio"]
