from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np

# read in "in vehicle coupon" database csv file and create a data frame
coupon_ = read_csv('in-vehicle-coupon-recommendation.csv')
original_index = coupon_.index
original_num_rows = len(original_index)  # get index number
# print('original number of rows ', original_num_rows)  # debug statement
# print((coupon_.info()))  # debug statement

# remove column with low number of entries
del coupon_['car']
# print((coupon_df.info()))  # debug statement

# how many null item rows are there?
# print('count of null item rows \n')  # debug statement
# print(coupon_.isnull().sum(), '\n')  # debug statement

# remove rows with null items
coupon_ = coupon_.dropna(axis=0)
# print('verify null rows removed \n')  # debug statement
# print(coupon_.isnull().sum(), '\n')  # debug statement

# reindex coupon_
coupon_.reset_index(inplace=True, drop=True)

# check coupon_ for new number of rows
# print('verify new number of rows changed \n')  # debug statement
# print((coupon_.info()))  # debug statement
reduced_index = coupon_.index
reduced_num_rows = len(reduced_index)  # get index number
# print('reduced number of rows ', reduced_num_rows)  # debug statement

# Step 1: how many entries where dropped
dropped_entries = original_num_rows - reduced_num_rows
print('Dropped', dropped_entries, 'entries')

# replace all strings with numerical values ###########################################################################
# print('Unreplaced Values List')  # debug statement
# for cols in coupon_:  # debug statement
#    print('\n', cols)  # debug statement
#    print(coupon_[cols].unique())  # debug statement

# Reference List of replaced values #########
# Destination Column
# 'No Urgent Place' = 0
# 'Home' = 1
# 'Work' = 2

# passanger column
# 'Alone'= 0
# 'Friend(s)' = 1
# 'Kid(s)' = 2
# 'Partner' = 3

# weather column
# 'Sunny' = 0
# 'Rainy' = 1
# 'Snowy' = 2

# time column
# '2PM' = 0
# '10AM' = 1
# '6PM' = 2
# '7AM' = 3
# '10PM' = 4

# coupon column
# 'Restaurant(<20)' = 0
# 'Coffee House' = 1
# 'Bar' = 2
# 'Carry out & Take away' = 3
# 'Restaurant(20-50)' = 4

# expiration column
# '1d' = 0
# '2h' = 1

# gender column
# 'Male' = 0
# 'Female' = 1

# age column
# '21' = 0
# '46' = 1
# '26' = 2
# '31' = 3
# '41' = 4
# '50plus' = 5
# '36' = 6
# 'below21' = 7

# maritalStatus column
# 'Single' = 0
# 'Married partner' = 1
# 'Unmarried partner' = 2
# 'Divorced' 3
# 'Widowed' = 4

# education column
# 'Bachelors degree' = 0
# 'Some college - no degree' = 1
# 'Associates degree' = 2
# 'High School Graduate' = 3
# 'Graduate degree (Masters or Doctorate)' = 4
# 'Some High School' = 5

# occupation column
# 'Architecture & Engineering' = 0
# 'Student' = 1
# 'Education&Training&Library' = 2
# 'Unemployed' = 3
# 'Healthcare Support' = 4
# 'Healthcare Practitioners & Technical' = 5
# 'Sales & Related' = 6
# 'Management' = 7
# 'Arts Design Entertainment Sports & Media' = 8
# 'Computer & Mathematical' = 9
# 'Life Physical Social Science' = 10
# 'Personal Care & Service' = 11
# 'Office & Administrative Support' = 12
# 'Construction & Extraction' = 13
# 'Legal' = 14
# 'Retired' = 15
# 'Community & Social Services' = 16
# 'Installation Maintenance & Repair' = 17
# 'Transportation & Material Moving' = 18
# 'Business & Financial' = 19
# 'Protective Service' = 20
# 'Food Preparation & Serving Related' = 21
# 'Production Occupations' = 22
# 'Building & Grounds Cleaning & Maintenance' = 23
# 'Farming Fishing & Forestry' = 24

# income column
# '$62500 - $74999' = 0
# '$12500 - $24999' = 1
# '$75000 - $87499' = 2
# '$50000 - $62499' = 3
# '$37500 - $49999' = 4
# '$25000 - $37499' = 5
# '$100000 or More' = 6
# '$87500 - $99999' = 7
# 'Less than $12500' = 8

# Bar column
# 'never' = 0
# 'less1' = 1
# '1~3' = 2
# 'gt8' = 3
# '4~8' = 4

# CoffeeHouse column
# 'less1' = 0
# '4~8' = 1
# '1~3' = 2
# 'gt8' = 3
# 'never' = 4

# CarryAway column
# '4~8' = 0
# '1~3' = 1
# 'gt8' = 2
# 'less1' = 3
# 'never' = 4

# RestaurantLessThan20 column
# '4~8' = 0
# '1~3' = 1
# 'less1' = 2
# 'gt8' = 3
# 'never' = 4

# Restaurant20To50 column
# 'less1' = 0
# 'never' = 1
# '1~3' = 2
# 'gt8' = 3
# '4~8' = 4

coupon_['destination'] = coupon_.replace(to_replace=['No Urgent Place', 'Home', 'Work'], value=[0, 1, 2])
coupon_['passanger'] = coupon_['passanger'].replace(to_replace=['Alone', 'Friend(s)', 'Kid(s)', 'Partner'], \
                                                    value=[0, 1, 2, 3])
coupon_['weather'] = coupon_['weather'].replace(to_replace=['Sunny', 'Rainy', 'Snowy'], value=[0, 1, 2])
coupon_['time'] = coupon_['time'].replace(to_replace=['2PM', '10AM', '6PM', '7AM', '10PM'], value=[0, 1, 2, 3, 4])
coupon_['coupon'] = coupon_['coupon'].replace(to_replace=['Restaurant(<20)', 'Coffee House', 'Bar', \
                                                          'Carry out & Take away', 'Restaurant(20-50)'], \
                                              value=[0, 1, 2, 3, 4])
coupon_['expiration'] = coupon_['expiration'].replace(to_replace=['1d', '2h'], value=[0, 1])
coupon_['gender'] = coupon_['gender'].replace(to_replace=['Male', 'Female'], value=[0, 1])
coupon_['age'] = coupon_['age'].replace(to_replace=['21', '46', '26', '31', '41', '50plus', '36', 'below21'], \
                                        value=[0, 1, 2, 3, 4, 5, 6, 7])
coupon_['maritalStatus'] = coupon_['maritalStatus'].replace(to_replace=['Single', 'Married partner', \
                                                                        'Unmarried partner', 'Divorced', 'Widowed'], \
                                                            value=[0, 1, 2, 3, 4])
coupon_['education'] = coupon_['education'].replace(to_replace=['Bachelors degree', 'Some college - no degree', \
                                                                'Associates degree', 'High School Graduate', \
                                                                'Graduate degree (Masters or Doctorate)', \
                                                                'Some High School'], value=[0, 1, 2, 3, 4, 5])
coupon_['occupation'] = coupon_['occupation'].replace(to_replace=['Architecture & Engineering', 'Student', \
                                                                  'Education&Training&Library', 'Unemployed', \
                                                                  'Healthcare Support', \
                                                                  'Healthcare Practitioners & Technical', \
                                                                  'Sales & Related', 'Management', \
                                                                  'Arts Design Entertainment Sports & Media', \
                                                                  'Computer & Mathematical', \
                                                                  'Life Physical Social Science', \
                                                                  'Personal Care & Service', \
                                                                  'Office & Administrative Support', \
                                                                  'Construction & Extraction', 'Legal', 'Retired', \
                                                                  'Community & Social Services', \
                                                                  'Installation Maintenance & Repair', \
                                                                  'Transportation & Material Moving', \
                                                                  'Business & Financial', \
                                                                  'Protective Service', \
                                                                  'Food Preparation & Serving Related', \
                                                                  'Production Occupations', \
                                                                  'Building & Grounds Cleaning & Maintenance', \
                                                                  'Farming Fishing & Forestry'], value=[0, 1, 2, 3, \
                                                      4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
                                                      21, 22, 23, 24])
coupon_['income'] = coupon_['income'].replace(to_replace=['$62500 - $74999', '$12500 - $24999', '$75000 - $87499', \
                                                          '$50000 - $62499', '$37500 - $49999', '$25000 - $37499', \
                                                          '$100000 or More', '$87500 - $99999', 'Less than $12500'], \
                                              value=[0, 1, 2, 3, 4, 5, 6, 7, 8])
coupon_['Bar'] = coupon_['Bar'].replace(to_replace=['never', 'less1', '1~3', 'gt8', '4~8'], value=[0, 1, 2, 3, 4])
coupon_['CoffeeHouse'] = coupon_['CoffeeHouse'].replace(to_replace=['less1', '4~8', '1~3', 'gt8', 'never'], \
                                                        value=[0, 1, 2, 3, 4])
coupon_['CarryAway'] = coupon_['CarryAway'].replace(to_replace=['4~8', '1~3', 'gt8', 'less1', 'never'], \
                                                    value=[0, 1, 2, 3, 4])
coupon_['RestaurantLessThan20'] = coupon_['RestaurantLessThan20'].replace(to_replace=['4~8', '1~3', 'less1', 'gt8', \
                                                                                      'never'], value=[0, 1, 2, 3, 4])
coupon_['Restaurant20To50'] = coupon_['Restaurant20To50'].replace(to_replace=['less1', 'never', '1~3', 'gt8', '4~8'], \
                                                                  value=[0, 1, 2, 3, 4])

# print('\n Replaced Values ListS')  # debug statement
# for cols in coupon_:  # debug statement
#    print('\n', cols)  # debug statement
#    print(coupon_[cols].unique())  # debug statement

# Step 3: Print top 10 correlations
corr = coupon_.corr().abs()
corr *= np.tri(*corr.values.shape, k=-1).T
corr = corr.unstack()
corr.sort_values(inplace=True, ascending=False)
print('\nTop Ten Correlations')
print(corr[:11])

# Step 4: Top Ten Correlations by Class
print('\nTop Ten Correlations by Class')
for cols in coupon_:
    corr_type= corr.get(key=cols)
    print('\n', cols)
    print(corr_type)  # couldn't print only first 10, error every way I tried

# Step 5: steps 3 and 4 but for Covariance
cov = coupon_.cov().abs()
cov *= np.tri(*cov.values.shape, k=-1).T
cov = cov.unstack()
cov.sort_values(inplace=True, ascending=False)
print('\nTop Ten Covariances')
print(cov[:11])
print('\nTop Ten Covariances by Class')
for cols in coupon_:
    cov_type= cov.get(key=cols)
    print('\n', cols)
    print(cov_type)  # couldn't print only first 10, error every way I tried

# Step 6: saves updated dataframe with new name coupon.csv
coupon = coupon_
coupon.to_csv('coupon.csv', index=False, header=True)




