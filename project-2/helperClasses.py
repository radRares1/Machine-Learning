# -*- coding: utf-8 -*-
"""
Created on Sat Nov 6 12:37:06 2021
@author: Rares2
"""

header = ["leftWeight", "leftDistance", "rightWeight", "rightDistance"]


def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = int(counts[label] / total * 100)

    # if len(probs) == 3:
    #     if probs['R'] == probs['L']:
    #         probs['R'] = 0
    #         probs['L'] = 0
    #         probs['B'] = 100
    return probs


def classCounts(rows):
    stats = {}

    for row in rows:
        classLabel = row[0]
        if classLabel not in stats:
            stats[classLabel] = 0
        stats[classLabel] += 1

    return stats


class Question:
    """
    helper class used to split the dataset
    """
    def __init__(self, attribute, value):
        self.col = attribute - 1
        self.val = value

    def answer(self, row):
        value = row[self.col]
        return value >= self.val

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.

        condition = ">="
        return "Is %s %s %s?" % (
            header[self.col], condition, str(self.val))


class Leaf:
    """
    helper class that clasifies data
    dictionary that holds the class label and its quantifier
    """

    def __init__(self, rows):
        self.predictions = classCounts(rows)


class DecisionNode:
    """
    helper class that asks a question
    has a ref to the question, and to the two answer-children nodes
    """

    def __init__(self, question, trueSide, falseSide):
        self.question = question
        self.trueSide = trueSide
        self.falseSide = falseSide
