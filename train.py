import numpy as np 
import math
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
import random 

from wordsegment import segment
from nltk import word_tokenize,pos_tag
import enchant
import nltk
import string
import csv

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re

from enchant.tokenize import get_tokenizer,HTMLChunker

df=pd.read_csv('train_indessa.csv')

labels_train=[]
features_train=[]
features_test=[]
labels_test=[]

LoanInfo={}
LoanStatus={}


cnt=1

for index,row in df.iterrows():
	print('{0}\r'.format(cnt),end='')
	cnt+=1

	LoanInfo[row['member_id']]=[]
	LoanInfo[row['member_id']].append(int(row['loan_amnt']))
	LoanInfo[row['member_id']].append(int(row['funded_amnt']))
	LoanInfo[row['member_id']].append(int(row['funded_amnt_inv']))
	LoanInfo[row['member_id']].append(int(row['term'][:-7]))
	LoanInfo[row['member_id']].append(int(row['int_rate']))
	grade=10*(ord(row['grade'])-64)+int(row['sub_grade'][1:])

	LoanInfo[row['member_id']].append(grade)

	# For emp_length, find a number inside string.
	emp_length=row['emp_length']
	items=re.findall('\d+',emp_length)
	if len(items)>0:
		emp_length=int(items[0])
	else:
		emp_length=0

	LoanInfo[row['member_id']].append(emp_length)

	# home_ownership
	# 1 - OWN , 2- RENT 3- MORTGAGE

	home_ownership=row['home_ownership']
	if home_ownership=="OWN":
		home_ownership=1
	elif home_ownership=="RENT":
		home_ownership=2
	elif home_ownership=="MORTGAGE":
		home_ownership=3
	else:
		home_ownership=0

	
	LoanInfo[row['member_id']].append(home_ownership)

	# Annual Inc and Verification Status
	annual_inc=row['annual_inc']
	if math.isnan(annual_inc):
		annual_inc=0

	status=row['verification_status']

	if status=="Source Verified":
		LoanInfo[row['member_id']].append(int(annual_inc))
	else:
		LoanInfo[row['member_id']].append(0)

	# pymnt_plan
	pymnt=row['pymnt_plan']
	if pymnt=="n":
		pymnt=0
	else:
		pymnt=1

	LoanInfo[row['member_id']].append(pymnt)

	# dti
	dti=row['dti']
	LoanInfo[row['member_id']].append(dti)

	#delinq_2yrs
	#Number of 30+ days delinquency in past 2 years
	if not math.isnan(row['delinq_2yrs']):
		delinq2yrs=int(row['delinq_2yrs'])
	else:
		delinq2yrs=0

	#delinq2yrs=int(row['delinq_2yrs'])
	LoanInfo[row['member_id']].append(delinq2yrs)

	#mths_since_last_delinq
	#print(row['mths_since_last_delinq'])
	if not math.isnan(row['mths_since_last_delinq']):
		mthslastdelinq=int(row['mths_since_last_delinq'])
	else:
		mthslastdelinq=10000


	LoanInfo[row['member_id']].append(mthslastdelinq)

	#mths_since_last_record
	if not math.isnan(row['mths_since_last_record']):
		mthssincelastrecord=int(row['mths_since_last_record'])
	else:
		mthssincelastrecord=10000

	LoanInfo[row['member_id']].append(mthssincelastrecord)
	#open_acc
	openacc=row['open_acc']

	if math.isnan(openacc):
		openacc=0
	else:
		openacc=int(openacc)

	LoanInfo[row['member_id']].append(openacc)
	#pub_rec
	pubrec=row['pub_rec']

	if math.isnan(pubrec):
		pubrec=0
	else:
		pubrec=int(pubrec)


	LoanInfo[row['member_id']].append(pubrec)

	revol_bal=row['revol_bal']

	if math.isnan(revol_bal):
		revol_bal=0
	else:
		revol_bal=revol_bal

	LoanInfo[row['member_id']].append(revol_bal)

	#revol_util
	revol_util=row['revol_util']

	if math.isnan(revol_util):
		revol_util=0
	else:
		revol_util=revol_util

	LoanInfo[row['member_id']].append(revol_util)

	#total_rec_int
	total_rec_int=row['total_rec_int']

	if math.isnan(total_rec_int):
		total_rec_int=0
	else:
		total_rec_int=total_rec_int

	LoanInfo[row['member_id']].append(total_rec_int)

	#total_rec_late_fee
	total_rec_late_fee=row['total_rec_late_fee']

	LoanInfo[row['member_id']].append(total_rec_late_fee)

	#mths_since_last_major_derog
	mths_since_last_major_derog=row['mths_since_last_major_derog']
	if math.isnan(mths_since_last_major_derog):
		mths_since_last_major_derog=10000
	else:
		mths_since_last_major_derog=int(mths_since_last_major_derog)

	LoanInfo[row['member_id']].append(mths_since_last_major_derog)

	#acc_now_delinq
	acc_now_delinq=row['acc_now_delinq']
	if math.isnan(acc_now_delinq):
		acc_now_delinq=0
	else:
		acc_now_delinq=int(acc_now_delinq)

	LoanInfo[row['member_id']].append(acc_now_delinq)

	#tot_cur_bal
	tot_cur_bal=row['tot_cur_bal']

	if math.isnan(tot_cur_bal):
		tot_cur_bal=0
	else:
		tot_cur_bal=tot_cur_bal

	LoanInfo[row['member_id']].append(tot_cur_bal)

	#loan_status 1=default, 0=non-defaulter
	LoanStatus[row['member_id']]=int(row['loan_status'])


keys=list(LoanStatus.keys())
random.shuffle(keys)

print("Preparing features_train and labels_train")


for key in keys:
	labels_train.append(LoanStatus[key])
	features_train.append(LoanInfo[key])

l2=int(0.67*len(features_train))
for i in range(l2+1,len(features_train)):
	features_test.append(features_train[i])

features_train=features_train[:l2+1]

l2=int(0.67*len(labels_train))

for i in range(l2+1,len(labels_train)):
	labels_test.append(labels_train[i])

labels_train=labels_train[:l2+1]

print(len(features_train),len(features_test),len(labels_train),len(labels_test))

from sklearn.metrics import roc_auc_score

print("Started training...")

"""
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
"""

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,verbose=5,n_jobs=-1)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

print(roc_auc_score(labels_test,pred))



















