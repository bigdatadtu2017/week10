# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:09:09 2017

@author: andre
"""

import json
from time import time
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mmh3         #has to be installed


percentage = 0.8    #Amount of training data
Estimators = 50     #Number of trees
bags = 1000         #Number of bags
find_this = 'earn'  #word used for prediction

path = "./mergedjson/merged.json"

##############################################
#Filter Json function
##############################################

def filter_json(path):
	filter_json = []
	with open(path) as json_file:
		json_data = json.load(json_file)

        #Remove all articles that do not have at least one topic and a body.
		json_data = filter(lambda x: "topics" in x.keys() and "body" in x.keys(), json_data)
		filter_json = json_data
	return filter_json

##############################################
#Filter Json function
##############################################

def MatrixDecleration(x,y):
	matrix = [[0]*y for i in range(x)]
	return matrix;

##############################################
#Bag of Words function
##############################################

def BagOfWords(merge_list):
	line_count = 0
	lines = []
	output = []
	word_uniq = []

	for json in merge_list:
		body = json["body"]	
		line_count += 1

        #Define output matrix
		if find_this in json["topics"]:
			output.append(1)
		else:
			output.append(0)

		      #find only unique words and in lower case
		text = set(re.findall('\w+', body.lower()))
		for word in text:
			if word in word_uniq:		
				next
			else:
				word_uniq.append(word)  
		lines.append(text)

    #Allocate using MatrixDecleration function
	bag_matrix = MatrixDecleration(line_count,len(word_uniq))
	bag_matrix_out = MatrixDecleration(line_count,2)
     
	for l in range(len(lines)):
            #fill in words
		for w in lines[l]:
			bag_matrix[l][word_uniq.index(w)] += 1
            #Make output matrix
		if output[l] == 1:
			bag_matrix_out[l][1] = 1
		else:
			bag_matrix_out[l][0] = 1

	print("Bag of Words Matrix Dimension: %d * %d" % (len(bag_matrix), len(bag_matrix[0])))
	print("Output Matrix Dimension : %d * %d" % (len(bag_matrix_out), len(bag_matrix_out[0])))

	result = []
	result.append(np.array(bag_matrix))
	result.append(np.array(bag_matrix_out))
	return result

##############################################
#Feature Hashing Function
##############################################

def FeatureHashing(filter_list,buckets):

	line_count = 0
	lines = []
	output = []

	for jsonn in filter_list:
		body = jsonn["body"]	
		line_count += 1

		if find_this in jsonn["topics"]:
			output.append(1)
		else:
			output.append(0)

    	#find only unique words and in lower case
		text = set(re.findall('\w+', body.lower()))
		indices = []
		for word in text:
            #Feature Hashing using mmh3 (Murmurhashing)
			idx = mmh3.hash(word.encode('utf-8')) % buckets
			indices.append(idx)

		lines.append(indices)


	hash_table = MatrixDecleration(line_count,buckets)
	hash_table_out = MatrixDecleration(line_count,2)

	for l in range(len(lines)):
        #fill in words
		for i in lines[l]:
			hash_table[l][i] += 1 	
            #make output matrix
		if output[l] == 1:
			hash_table_out[l][1] = 1
		else:
			hash_table_out[l][0] = 1

	print("Hashing - Bag Of Words Dimension : %d * %d" % (len(hash_table), len(hash_table[0])))
	print("Hashing - Matrix Dimension: %d * %d" % (len(hash_table_out), len(hash_table_out[0])))

	result = []
	result.append(np.array(hash_table))
	result.append(np.array(hash_table_out))
	return result

##############################################
#Random Forrest Classifier function
##############################################

def RandomForest(data):
    #inputs either bag of words or feature hashing
	xData = data[0]
	yData = data[1]
	yData_Vec = []

	for i in yData:
		if i[0] == 1:
			yData_Vec.append(0)
		else:
			yData_Vec.append(1)

    #selecting 80% of data as training sample
	xTrain = np.array( xData[:round(xData.shape[0] * percentage)] )
	yTrain = np.array( yData_Vec[:int( round(percentage * len(yData_Vec)) )] )
    #selection 20% of data as test sample
	xTest = np.array( xData[-round(xData.shape[0] * (1.0-percentage)):] )
	yTest = np.array( yData_Vec[-int( round((1.0-percentage) * len(yData_Vec)) ):] )
    
    #Build model with number of estimators defined
	forestModel = RandomForestClassifier(Estimators)
    #Train model using train data
	forestModel.fit(xTrain, yTrain)
    #Find accuracy of classifier using test set
	print ("Classifier Accuracy %f in percent" % (forestModel.score(xTest, yTest)*100) )

##############################################
#Use functions and print result
##############################################

print('---------Random Forest Bag of Words----------------')
start_time = time()
#Use functions and output Bag-of-Words and Output Dimensions, and Classifier accuracy
RandomForest(BagOfWords(filter_json(path))) 
print("Run Time:  %s [s] " % (time() - start_time))

start_time = time()
print('--------Random Forest Feature Hash-----------------')
#Use functions and output Bag-of-Words and Output Dimensions, and Classifier accuracy
RandomForest(FeatureHashing( filter_json(path),bags ))
print("Run Time:  %s [s] " % (time() - start_time))

#%%







