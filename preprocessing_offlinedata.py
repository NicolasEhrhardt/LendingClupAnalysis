import pandas as pd
import numpy as np
import re

FILENAMES = (
    'data/LoanStats3c_securev1.csv',
    'data/LoanStats3d_securev1.csv',
)


LOAN_STATUS_COL = "loan_status"
Y_COL = "Y"
BAD_FLAGS = ['Charged Off', 'Default']


ALL_COLS = {
    "id",
    "member_id",
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "pymnt_plan",
    "url",
    "desc",
    "purpose",
    "title",
    "zip_code",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "policy_code"
}


TRAINING_COLS = [
    'zip_code',
    'addr_state',
    'annual_inc',
    'collection_recovery_fee',
    'collections_12_mths_ex_med',
    'delinq_2yrs',
    'desc',
    'dti',
    #'earliest_cr_line',
    'emp_length',
    #'emp_title',
    'fico_range_high',
    'fico_range_low',
    'funded_amnt',
    'funded_amnt_inv',
    'term',
    'grade',
    'sub_grade',
    'home_ownership',
    'initial_list_status',
    'inq_last_6mths',
    'installment',
    'int_rate',
    #'issue_d',
    #'last_credit_pull_d',
    #'last_fico_range_high',
    #'last_fico_range_low',
    #'last_pymnt_amnt',
    #'last_pymnt_d',
    'loan_amnt',
]


DATE_COLS = [
    #'earliest_cr_line',
    #'issue_d',
    #'last_credit_pull_d',
    #'last_pymnt_d',
 ]


CATEGORY_COLS = {
    #'emp_title': False,
    #'emp_length': True,
    'zip_code': False,
    'addr_state': False,
    'term': True,
    'grade': True,
    'sub_grade': True,
    'home_ownership': False,
    'initial_list_status': False,
}


def get_data(file_names=FILENAMES):
    data_chunks = [
        pd.read_csv(file_name, low_memory=False, skiprows=1, na_values='n/a')
        for file_name in file_names
    ]

    data = pd.concat(data_chunks)

    # Clean-up
    data.dropna(axis=0, subset=[LOAN_STATUS_COL], inplace=True)

    for date_col in DATE_COLS:
        print('Casting [%s] to datetime' % (date_col,))
        data[date_col] = pd.to_datetime(data[date_col], format='%b-%Y')

    for cat_col, ordered in CATEGORY_COLS.items():
        print('Casting [%s] to category' % (cat_col,))
        data[cat_col] = pd.Categorical(data[cat_col], ordered=ordered)

    # Special treatments...
    # Id as int
    data['id'] = data['id'].astype('int64')

    # Int rate as float
    data['int_rate'] = data['int_rate'].apply(lambda x: x[0:-2] if x is not np.nan else np.nan).astype(float)

    # Employment
    # n/a in length
    num_re = re.compile('[0-9]*')
    data['emp_length'] = data['emp_length'].apply(lambda x: 0 if x is np.nan else '0' + re.match(num_re, x).group(0)).astype(float)

    # Description
    data['desc'] = data['desc'].isnull()

    return data


def filter_data(data):
    return data[TRAINING_COLS + [LOAN_STATUS_COL]]


def dummify(data, columns=CATEGORY_COLS):
    new_data = data
    dummies = []

    for cat_col in columns:
        print ('Dummifying [%s]' % (cat_col,))
        dummies.append(pd.get_dummies(data[cat_col]))
        new_data.drop(cat_col, axis=1, inplace=True)

    new_data = pd.concat([new_data] + dummies, axis=1)
    
    return new_data


def get_training_data(file_names=FILENAMES):
    data_raw = get_data(file_names)
    data_filtered = filter_data(data_raw)
    data = dummify(data_filtered)

    training_data = data.drop(LOAN_STATUS_COL, axis=1)
    training_labels = data[LOAN_STATUS_COL]
    training_binary = training_labels.apply(lambda x: x not in BAD_FLAGS)

    return training_data, training_binary
