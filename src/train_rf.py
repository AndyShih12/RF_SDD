from __future__ import with_statement
#/usr/bin/env jython

import os, sys, argparse

path = os.path.realpath(__file__)
path = os.path.dirname(path)
path1 = os.path.join(path,'../lib')
path1 = os.path.join(path1,'weka.jar')
if path1 not in sys.path: sys.path.append(path1)

from utils import timer
from utils import readConfig


import weka.classifiers.trees.RandomTree as RandomTree
import weka.classifiers.trees.RandomForest as RandomForest
import weka.classifiers.bayes.NaiveBayes as NaiveBayes

import weka.core.converters.ConverterUtils.DataSource as DataSource
import weka.filters.unsupervised.attribute.NumericToNominal as NumericToNominal
import weka.filters.Filter as Filter
import java.lang.StringBuffer as StringBuffer
import weka.classifiers.Evaluation as Evaluation
import weka.classifiers.evaluation.output.prediction.PlainText as PlainText

def read_dataset(filename,class_index=0):
    data = DataSource(filename).getDataSet()
    #convert = NumericToNominal()
    #convert.setInputFormat(data)
    #data = Filter.useFilter(data,convert)
    data.setClassIndex(class_index)
    return data

def evaluate_dataset(classifier,data):
    evaluation = Evaluation(data)
    output = PlainText()
    output.setHeader(data)
    eval_buffer = StringBuffer() # buffer to use
    output.setBuffer(eval_buffer)
    options = [output]
    evaluation.evaluateModel(classifier,data,options)
    return evaluation


def binarize_helper_node_list(node, tree):
    if node.m_Attribute == -1:
      return []
    ret = [(tree.m_Info.attribute(node.m_Attribute).name(), node.m_SplitPoint)]
    for ch in node.m_Successors:
      ret = ret + binarize_helper_node_list(ch,tree)
    return ret

def binarize(tree):
    nodes = binarize_helper_node_list(tree.m_Tree, tree)
    nodes.sort()

    print nodes


def run(basename,train_filename,test_filename,
        num_trees=100,tree_depth=0,class_index=0):

    with timer.Timer("loading data"):
        training = read_dataset(train_filename,class_index=class_index)
        testing = read_dataset(test_filename,class_index=class_index)

    """
    print "====== naive Bayes ====="
    with timer.Timer("training"):
        nb = NaiveBayes()
        nb.buildClassifier(training)
    with timer.Timer("testing"):
        eval_training = evaluate_dataset(nb,training)
        eval_testing = evaluate_dataset(nb,testing)
    print "=== evaluation (training):"
    print eval_training.toSummaryString()
    print "=== evaluation (testing):"
    print eval_testing.toSummaryString()
    """

    print "====== random forest ====="
    with timer.Timer("training"):
        rf = RandomForest()
        #rf.setOptions([
        #  u'-P', u'100', u'-I', u'100', u'-num-slots', u'1', u'-K', u'0', u'-M', u'1.0', u'-V', u'0.001', u'-S', u'1',
        #  u'-num-decimal-places', u'6'
        #])
        rf.setNumIterations(num_trees)
        if tree_depth:
            rf.setMaxDepth(tree_depth)
        rf.buildClassifier(training)
    with timer.Timer("testing"):
        eval_training = evaluate_dataset(rf,training)
        eval_testing = evaluate_dataset(rf,testing)
    print "=== evaluation (training):"
    print eval_training.toSummaryString()
    print "=== evaluation (testing):"
    print eval_testing.toSummaryString()

    #print rf.getmembers()

    num_classifiers = len(rf.m_Classifiers)
    for i,tree in enumerate(rf.m_Classifiers):
        options_arr = tree.getOptions()
        options_arr_python = [x for x in options_arr]
        options_arr_python += [u'-num-decimal-places',u'6']
        tree.setOptions(options_arr_python)
        #print tree.toString()
        #binarize(tree)
        filename = basename % i
        with open(filename,"w") as f:
            f.writelines(tree.graph())

    correct,incorrect = 0,0
    for instance in testing:
        pos,neg = 0,0
        for tree in rf.m_Classifiers:
            #print tree.classifyInstance(instance)
            if tree.classifyInstance(instance) >= 0.5:
                pos += 1
            else:
                neg += 1
            my_label = 1.0 if pos >= neg else 0.0
        if my_label == instance.classValue():
            correct += 1
        else:
            incorrect += 1
    print "    trees : %d" % num_trees
    print "--- evaluating majority vote on random forest:"
    print "  correct : %d" % correct
    print "incorrect : %d" % incorrect

parser = argparse.ArgumentParser('Trains a random forest from data.')

parser.add_argument('config_file', type=str, help='The config file.')
args = parser.parse_args()

if __name__ == '__main__':
    config_file = args.config_file
    config_json = readConfig.read_json(config_file)

    dataset = config_json["dataset"]
    basename = config_json["rf_basename"] 
    train_filename = config_json["train_filename"]
    test_filename = config_json["test_filename"]
    num_trees = config_json["tree_count"]
    tree_depth = config_json["tree_depth"]

    run(basename=basename,
        train_filename=train_filename,
        test_filename=test_filename,
        num_trees=num_trees,
        tree_depth=tree_depth,
        class_index=0)
