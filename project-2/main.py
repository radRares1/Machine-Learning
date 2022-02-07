# -*- coding: utf-8 -*-
"""
Created on Sat Nov 6 10:58:31 2021

@author: Rares2
"""
import random

from helperClasses import *
import operator

# lab6 decision trees

def partition(rows, question):
    """
    splits the data set into paritions
    we split the data by answering the question(i.e: >=3)
    """
    trueRows = []
    falseRows = []
    for row in rows:
        if question.answer(row):
            trueRows.append(row)
        else:
            falseRows.append(row)

    return trueRows, falseRows

def giniIndex(rows):
    """
    get the giniIndex for a list of rows
    """
    stats = classCounts(rows)
    impurity = -1
    for label in stats:
        labelProb = stats[label] / float(len(rows))
        impurity -= labelProb ** 2
    return impurity

def infoGain(left, right, currentUncertainty):
    """
    uncertainty of starting node - weighted avg impurity of
    child nodes
    """

    weight = float(len(left)) / (len(left) + len(right))

    return currentUncertainty - weight * giniIndex(left) - (1 - weight) * giniIndex(right)

def getBestSplit(rows):
    """
    get the best question by calculating the info gain of
    all values of each attribute
    """
    bestGain = 0
    bestQuestion = None

    currentUncertainty = giniIndex(rows)

    for attribute in range(1, len(rows[0])):

        # get all the values of that attribute
        values = set([row[attribute] for row in rows])

        # test all the values for the best split
        for value in values:

            question = Question(attribute, value)

            trueRows, falseRows = partition(rows, question)

            # skip it if it won;t split
            if len(trueRows) == 0 or len(falseRows) == 0:
                continue

            # get the info gain of this split

            gain = infoGain(trueRows, falseRows, currentUncertainty)

            # print(gain)

            # update the bestGain if better

            if gain >= bestGain:
                bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

def makeTree(rows):
    """
    we build the tree using recursion
    we perform the partitioning,
    get the info gain for each split,
    return the question with the best gain
    """

    gain, question = getBestSplit(rows)

    # we stop if the gain is 0
    if gain == 0:
        return Leaf(rows)

    # partition the data by answer to the question
    trueRows, falseRows = partition(rows, question)

    # start recursion for true and false child-trees
    trueBranch = makeTree(trueRows)

    falseBranch = makeTree(falseRows)

    return DecisionNode(question, trueBranch, falseBranch)

def printTree(node, string=""):
    if isinstance(node, Leaf):
        print(string + "prediction:", node.predictions)
        return

    print(string + str(node.question))

    print(string + "True branch:")
    printTree(node.trueSide, string + " ")

    print("\n")

    print(string + "False branch:")
    printTree(node.falseSide, string + " ")

def classification(row, node):
    """
    again we go recursively through the built tree
    """

    if isinstance(node, Leaf):
        return node.predictions

    # answer the question and go on the associated branch

    if node.question.answer(row):
        return classification(row, node.trueSide)
    else:
        return classification(row, node.falseSide)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def flatten(t):
    return [item for sublist in t for item in sublist]

def k_fold(k, data):
    """
    splits the data and performs a k-fold cross validation of the model
    :param k: number of folds
    :param data: list of rows to be split
    :return: returns the averages of average measurements.
    """
    measuresList = []
    for i in range(0, k):
        print(i)
        splitData = list(split(data, k + 1))
        testData = splitData[i]
        dataWithoutTest = splitData.copy()
        dataWithoutTest.pop(i)
        trainData = flatten(dataWithoutTest)
        tree = train(trainData)
        currentMeasures = test(testData, tree)
        print(currentMeasures)
        measuresList.append(currentMeasures)

    evaluationMap = averageTheEvaluation(measuresList)
    return evaluationMap

def averageAll(measurements, measure):
    total = 0
    for item in measurements:
        total += item[measure]

    return total/len(measurements)

def averageTheEvaluation(measurements):
    evaluations = {}
    accuracy = averageAll(measurements,"accuracy")
    precision = averageAll(measurements,"precision")
    specificity = averageAll(measurements,"specificity")
    recall = averageAll(measurements,"recall")
    fMeasure = averageAll(measurements,"fMeasure")

    evaluations["accuracy"] = accuracy
    evaluations["precision"] = precision
    evaluations["specificity"] = specificity
    evaluations["recall"] = recall
    evaluations["fMeasure"] = fMeasure

    return evaluations

def train(data):
    """
    creates and prints the tree
    :param data: list of rows
    :return: the tree
    """
    tree = makeTree(data)
    #printTree(tree)
    return tree

def computeConfusion(data, label, tree):
    truePos = 1
    trueNeg = 1
    falsePos = 1
    falseNeg = 1
    for row in data:
        d = printLeaf(classification(row, tree))
        # print("Actual: %s. Predicted: %s" % (row[0], d))

        letter = max(d.items(), key=operator.itemgetter(1))[0]

        if row[0] == letter and row[0] == label:
            truePos += 1
        if row[0] != letter and row[0] != label:
            trueNeg += 1
        if row[0] == letter and row[0] != label:
            falsePos += 1
        if row[0] != letter and row[0] == label:
            falseNeg += 1

    confusion = {}
    confusion["truePos"] = truePos
    confusion["trueNeg"] = trueNeg
    confusion["falsePos"] = falsePos
    confusion["falseNeg"] = falseNeg

    return confusion

def computeMeasures(confusion):
    measures = {}
    truePos = confusion["truePos"]
    trueNeg = confusion["trueNeg"]
    falsePos = confusion["falsePos"]
    falseNeg = confusion["falseNeg"]
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falseNeg + falsePos)
    precision = truePos / (truePos + falsePos)
    specificity = trueNeg / (trueNeg + falsePos)
    recall = truePos / (truePos + falseNeg)
    fMeasure = truePos/ (truePos + 1/2 * (falsePos + falseNeg))

    measures["accuracy"] = accuracy
    measures["precision"] = precision
    measures["specificity"] = specificity
    measures["recall"] = recall
    measures["fMeasure"] = fMeasure

    return measures

def test(data, tree):

    evaluationL = computeMeasures(computeConfusion(data, 'L', tree))
    evaluationB = computeMeasures(computeConfusion(data, 'B', tree))
    evaluationR = computeMeasures(computeConfusion(data, 'R', tree))

    lenL = len(list(filter(lambda x: (x[0] == "L"), data)))
    lenB = len(list(filter(lambda x: (x[0] == "B"), data)))
    lenR = len(list(filter(lambda x: (x[0] == "R"), data)))

    measures={}

    averageAccuracy = (evaluationR["accuracy"]*lenR + evaluationB["accuracy"]*lenB + evaluationL["accuracy"]*lenL)/len(data)
    averagePrecision = (evaluationR["precision"]*lenR + evaluationB["precision"]*lenB + evaluationL["precision"]*lenL)/len(data)
    averageSpecificiy = (evaluationR["specificity"]*lenR + evaluationB["specificity"]*lenB + evaluationL["specificity"]*lenL)/len(data)
    averageRecall = (evaluationR["recall"]*lenR + evaluationB["recall"]*lenB + evaluationL["recall"]*lenL)/len(data)
    averagefMeasure = (evaluationR["fMeasure"]*lenR + evaluationB["fMeasure"]*lenB + evaluationL["fMeasure"]*lenL)/len(data)

    measures["accuracy"] = averageAccuracy
    measures["precision"] = averagePrecision
    measures["specificity"] = averageSpecificiy
    measures["recall"] = averageRecall
    measures["fMeasure"] = averagefMeasure


    return measures

if __name__ == '__main__':

    file = open("balance-scale.data", "r")

    lines = file.readlines()

    data = []

    for line in lines:
        newList = []
        for item in line:
            if item != "," and item != "\n":
                newList.append(item)
        data.append(newList)

    print("size of data \n")
    print(len(data))

    #random.shuffle(data) - nvm, not good

    evaluations = k_fold(100,data)
    print(evaluations)

    from math import sqrt
    interval = 1.96 * sqrt( (evaluations["accuracy"] * (1 - evaluations["accuracy"])) / 50)
    print("confidence interval")
    print('%.3f' % interval)
