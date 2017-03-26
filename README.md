# BankLoanliness
My solution for ML challenge #1 Hackerearth. https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-one/machine-learning/bank-fears-loanliness/

~~Currently I am using Random Forest Classifier with 66:34 split in training and testing dataset. Getting 65 % accuracy.~~

I am using following features :- 

1) Loan amount

2) TErm of Loan

3) Int Rate

4) 10*grade+subgrade

5) emp_length

6) home_ownership

7) annual_inc+verification status

8) pymnt plan

9) dti

10) delinq_2yrs

11) mths_since_last_delinq

12) mths_since_last_record

13) open_acc

14) pubrec

15) revol_bal

16) revol_util

17) total_rec_int

18) total_rec_late_fee

19) mths_since_last_major_derog

20) last_week_pay

21) acc_now_delinq

22) tot_cur_bal

Using Random Forest with n_estimators=100,verbose=5,n_jobs=-1, accuracy is 69%.



