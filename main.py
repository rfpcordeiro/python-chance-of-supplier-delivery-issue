import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime
#SEM DAG
try:
    from connectors.connector_gbq import ConnectorGBQ
#COM DAG
except ImportError:
    from dependencies.libs.common.connector.connectors.connector_gbq import ConnectorGBQ
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
# from matplotlib.cbook import boxplot_stats
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.decomposition import PCA

from sklearn import metrics
#SEM DAG
try:
    import constants as c
#COM DAG
except ImportError:
    import dependencies.libs.chnc_dlvry_issu.constants as c

def get_input_path(file_name):
    """
    Get the standard input file path of the project
    
    Parameters
    ----------
    file_name : str
        The file name which you want to join with the path

    Returns
    -------
    full_path : str
        Full standard input file path of the project
    """
    full_path = os.path.join(os.path.dirname(os.path.abspath("")), "input", file_name)
    return full_path

def get_output_path(file_name, p_type, model_name):
    """
    Get the standard input file path of the project
    
    Parameters
    ----------
    file_name : str
        The file name which you want to join with the path
    p_type : str
        If equals "DAG" choose the path used in the bucket

    Returns
    -------
    full_path : str
        Full standard input file path of the project
    """
    if p_type == "DAG":
        if model_name == 'canc_deliv_gnb':
            nm = 'canc_problty'
        if model_name == 'late_deliv_gnb':
            nm = 'late_problty'
        full_path = os.path.join(
            '/home/airflow/gcs/data/chnc_dlvry_issu/' + nm + '/models',
            file_name)
    else:
        full_path = os.path.join(os.path.dirname(os.path.abspath("")), "output", file_name)
    return full_path

def one_hot_encoder(df, lst_cat_cols):
    """
    Aplly one hot enconding in a column of a data frame
    
    Parameters
    ----------
    df : pandas.DataFrame
        data frame you want to apply the OHE
    lst_cat_cols : list
        list with column names you want to apply the OHE
    Returns
    -------
    df_res : pandas.DataFrame
        data frame after applying OHE
    """
    print(f'{time.ctime()}, Start of One Hot Enconding process')
    for col_nm in lst_cat_cols:
        print(f'{time.ctime()}, Analysing column: {col_nm}')
        # create a list with the original data frame and another dataframe with the 
        # columns created by yhe OHE
        lst_result = [df, pd.get_dummies(df[col_nm], prefix=col_nm)]
        print(f'{time.ctime()}, New columns created')
        # concatenate both data frames
        df = pd.concat(lst_result, axis=1, sort=False)
        print(f'{time.ctime()}, New columns merged to original data frame')
        # delete the columns used in OHE
    for col_nm in lst_cat_cols:
        del df[col_nm]
        print(f'{time.ctime()}, Column {col_nm} deleted')
    return df


def scale_data_frame(df, scl_typ='StandardScaler'):
    """
    Apply standard scaling to a data frame and remove the old column value.
    Do this process for each column of the list
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data frame you want to scale
    lst : list
        List with numerical columns of the data that you want to scale
        
    Results
    -------
    df : pandas.DataFrame
        Your scaled data frame
    """
    if scl_typ=='StandardScaler':
        # instanciates a variable with a standard scaler object
        scaler = StandardScaler()
    elif scl_typ=='MinMaxScaler':
        scaler = MinMaxScaler()
    elif scl_typ=='MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scl_typ=='RobustScaler':
        scaler = RobustScaler()
    elif scl_typ=='PowerTransformer':
        scaler = PowerTransformer()
    elif scl_typ=='QuantileTransformer':
        scaler = QuantileTransformer()
    print(f'{time.ctime()}, Scaler choosed: {scl_typ}')    
    lst = [col for col in df.columns if col not in c.lst_cat_cols and col != 'TRGT']
    # for each numerical columns of the that frame
    for col in df[lst].columns:
        print(f'{time.ctime()}, Scaling process of column {col} started')
        # fit the scaler with a column
        scaler.fit(df[[col]])
        print(f'{time.ctime()}, Scaler fitted')
        # transform the data of the column and save it overwritting the old values
        df[col] = scaler.transform(df[[col]])
        print(f'{time.ctime()}, Column scaled')
    # after all columns have been scaled return the data frame
    return df

def classifier_accuracy(y_test, y_pred):
    """
    Check the classification model accuracy.
    This is going to plot the confusion matrix and the area under the curve
    
    Parameters
    ----------
    y_test : list
        List with test values from sklearn.split_test
    y_pred : list
        List with classifier result
        
    Results
    -------
    None
    """
    y_test_flat = []
    for sublist in y_test.to_numpy():
        for item in sublist:
            y_test_flat.append(item)
    print(f'confusion_matrix: ')
    print(metrics.confusion_matrix(y_test_flat, y_pred))
    print('classification_report \n\n', metrics.classification_report(y_test_flat, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test_flat, y_pred)
    print(f'auc: {round(metrics.auc(fpr, tpr),2)}')
    
def create_group_df(df, lst_num_cols, cat_col_nm, lst_agg_mthds):
    """
    Add aggregated values to a data frame
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with all data
    lst_num_cols : list
        List of numerical columns which you want to use in your aggregation
    cat_col_nm : str
        name of the categorical column which is going to be aggregated
    lst_agg_mthds : list
        List of which mathematical aggregation you want to apply (max, min, average, standard deviation, etc)
        
    Results
    -------
    df_aux : pandas.DataFrame
        Data frame with all data, with aggregated values added
    """
    print(f'{time.ctime()}, Start groupping data based on column {cat_col_nm}')
    # append the name of the categorical columns which the data frame is going to be grouped by
    # otherwise the method will not find it
    lst_num_cols.append(cat_col_nm)
    # create a auxliar data frame, which groups columns by the desired 
    # categorical column, and finds the desired mathematical methods
    df_aux = df[lst_num_cols].groupby([cat_col_nm]).agg(lst_agg_mthds)
    print(f'{time.ctime()}, Grouped data frame created')
    # create an empty list which is going to receive the correted names of the columns
    lst_rnm = []
    for col in df_aux.columns:
        # append to the list the correct version of the column name
        lst_rnm.append(col[0]+'_'+col[1].upper()+"_"+cat_col_nm)
    print(f'{time.ctime()}, New column names defined')
    # how the final data frame has a multi-level header we need to drop it
    df_aux.columns = df_aux.columns.droplevel()
    # and at last, rename each column
    df_aux.columns = lst_rnm
    print(f'{time.ctime()}, Columns renamed')
    return df_aux

def prep_data(df, dict_dfs):
    """
    Apply all needed treatments to the data frames so they can be used on training and prediction
    
    Parameters
    ----------
    df_issue : pandas.DataFrame
        Data frame with the problematic deliveries
    df_on_time : pandas.DataFrame
        Data frame with deliveries which everything went well
    dict_dfs : list
        Dictionary with groupped data from categorical columns
        Follows this template -> {'categorical_column_name': 'df_with_groupped_data',}
    
    Results
    -------
    df : pandas.DataFrame
        Data frame after all treatments done to it
    """
    print(f'{time.ctime()}, Start of data treatment')
    # calculate the percentage of lead time which has already passed
    df['TIME_SPNT_PRCNTG'] = df['DAYS_OPEN']/df['PROMISED_LEAD_TIME'] # ta aqui
    print(f'{time.ctime()}, TIME_SPNT_PRCNTG calculated')
    # change data types to string, so the data frames are able of merging
    for col in c.lst_group_cols:
        df[col] = df[col].astype(str) # ta aqui
    print(f'{time.ctime()}, Changed data type of columns which has groupped values to be added')
    # merge the values to the main data frame
    for col in dict_dfs:
        df = df.merge(dict_dfs[col], how="left", on=[col]).set_index(df.index) # ta aqui
        print(f'{time.ctime()}, {col} grouped data added')
    # exclude the vendor number, it is used only as a key on the merge process
    df = df.drop(labels='VEND_NR', axis=1) # ta aqui
    # convert the date columns to weekdays, this way we could apply OHE to it
    for col in ['SCHDL_PLAN_DVSN_DCMNT_DT','PRCHS_DCMNT_DT']:
        df[col] = df[col].apply(lambda x: x.weekday()) # ta aqui
    print(f'{time.ctime()}, Weekdays calculated')
    # create a list with all numerical columns
    # do not consider the Target (TRGT) column and all columns from HIERCY_NODE_2 group
    lst = [col for col in df.columns if col not in c.lst_cat_cols and col != 'TRGT' and col.find("_HIERCY_NODE_2") == -1]
    for col in lst:
        # once we are talking about production we could not exclude values because we need to see them
        print(f'{time.ctime()}, Treating column {col}, rows qty {len(df)}')
        # to not exclude the 0 erros we can just chance them to a really low value
        df[col] = df[col].apply(lambda x: x if x>0 else 0.001) # ta aqui
        print(f'{time.ctime()}, Corrected values lower than 0')
        # the same analogy could be applied to the infinite values
        df[col] = df[col].apply(lambda x: 10^9 if x==float('inf') else x) # ta aqui
        print(f'{time.ctime()}, Corrected infinite values')
        # onde these two problems are solved we can apply log of n to the column
        # otherwise it would reaturn an error
        df[col] = df[col].apply(lambda x: np.log(x)) # ta aqui
        print(f'{time.ctime()}, Applied ln() to the values of the column')
    print(f'{time.ctime()}, All columns were treated')
    return df

def get_cumulative_lst(lst):
    """
    Apply Pareto principle to a list.
        
    Parameters
    ----------
    lst : list
        List of numbers
        
    Results
    -------
    lst_cumulative : lst
        List after Pareto principle
    """
    lst_cumulative = []
    # for each component of the data frame
    for i in range(len(lst)):
        # if the list is not empty
        if len(lst_cumulative):
            # append to the list the sum of the atual component and the cumulative value
            lst_cumulative.append(lst_cumulative[i-1] + lst[i])
        # if the list is empty
        else:
            # append to the list the atual component value
            lst_cumulative.append(lst[i])
    return lst_cumulative

def pca_optimal_n_components(X, opt_comp_ind=True, cmltv_thrshld=0.85, fig_sz_x=20, fig_sz_y=8):
    """
    Function that helps you understand the ideal number of components to be used in your PCA.
    
    Calculate, if opt_comp_ind equals True, the optimal number of components to your PCA, based on the cumulative threshold value informed.
        
    Parameters
    ----------
    X : list
        List with the variance ratio for each component (pca.explained_variance_ratio_)
    opt_comp_ind : boolean
        Optimal component inficator.
        If set as True, the function returns the optimal number of components for your PCA based on the cumulative threshold value
    cmltv_thrshld : float
        Cumulative threshold value, must be between 0 and 1.
    fig_sz_x : int
        Figure X axis size
    fig_sz_y : int
        Figure Y axis size
        
    Results
    -------
    opt_comp_num : int
        The number of components that should be used in your PCA analysis
    """
    pca = PCA(n_components=len(X.columns)-1)
    print(f'{time.ctime()}, PCA calculated using all columns')
    pca.fit(X)
    print(f'{time.ctime()}, PCA fitted')
    x_pca = pca.transform(X)
    print(f'{time.ctime()}, X trasformed calculated')
    plt.figure(figsize=(fig_sz_x,fig_sz_y))
    sns.heatmap(
        pd.DataFrame(pca.components_, columns=X.columns),
        cmap='plasma')
    # create the figure and the axis
    fig1 = plt.figure(figsize=(fig_sz_x,fig_sz_y))
    ax1 = fig1.add_subplot(111)
    # define x axis
    x = range(1, len(pca.explained_variance_ratio_)+1)
    # define array which represents the cumulative value as number of components increases
    y_cumulative = get_cumulative_lst(pca.explained_variance_ratio_)
    print(f'{time.ctime()}, Pareto calculated')
    # plot both arrays
    plt.plot(x,pca.explained_variance_ratio_, marker='o', )
    plt.plot(x,y_cumulative, marker='v', )
    # add annotations to the cumulative plot
    for i,j in zip(x,y_cumulative):
        ax1.annotate(str(round(j, 2)),xy=(i,j))
    # show all component numbers on X axis
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.grid(axis='x')
    plt.xlabel(xlabel='Number of Principal Components')
    plt.ylabel(ylabel='Proportion of Variance')
    if opt_comp_ind:
        # create a list with the values that are bigger than cmltv_thrshld
        # how you want the first bigger value, select the index 0 if it
        # now you have the first bigger value, it's time to apply lst.index() to get the respective index
        # but how indexes start from 0 and our X axis start from 1, add one to the value
        opt_comp_num = y_cumulative.index([v for v in y_cumulative if v>=cmltv_thrshld][0])+1
        print(f'{time.ctime()}, Optimal number of components found')
        # plot a vertical red line on the optimal components number
        plt.axvline(x=opt_comp_num, c='r')
        # plot a horizontal black line on the cmltv_thrshld value
        plt.axhline(y=cmltv_thrshld, c='k')
        # add legend to the plot
        plt.legend(labels=['Per Component', 'Cumulative', f'Optimal Component: {opt_comp_num}', f'Cumulative Threshold: {round(cmltv_thrshld,2)}'])
        return opt_comp_num
    else:
        # add legend to the plot
        plt.legend(labels=['Per Component', 'Cumulative'])    
        
def get_groupped_dfs():
    """
    Download from Google Big Query all groupped data frames
    
    Parameters
    ----------
    None
    
    Results
    -------
    dict_dfs : dict
        Dictionary of data frames, where the key of the dict is the column used as base of group by function in SQL query
        template -> 'column name' : data_frame
    """
    # instantiates connector GBQ to a variable
    connector = ConnectorGBQ()
    # instantiates an empty data dictionary
    dict_dfs = {}
    # for each column name in the list
    for col in c.lst_group_cols:
        # get the data from big query and insert it to the data frame
        dict_dfs[col] = connector.read_query(c.q_global_deliv.format(cat_col=col))
        print(f'{time.ctime()}, Groupped data based on {col} read')
    return dict_dfs
        
def get_data_train():
    """
    Download from Google Big Query the needed data to update/retrain the classification model
    
    Parameters
    ----------
    None
    
    Results
    -------
    df_canc : pandas.DataFrame
        Data frame with canceled deliveries (status 90 and 91)
    df_late : pandas.DataFrame
        Data frame with late deliveries (status 40 and 60), and PSTNG_DT (receiving date) > SCHDL_PLAN_DVSN_DCMNT_DT (promissed delivery date)
    df_on_time : pandas.DataFrame
        Data frame with late deliveries (status 40 and 60), and PSTNG_DT (receiving date) <= SCHDL_PLAN_DVSN_DCMNT_DT (promissed delivery date)
    dict_dfs : dict
        Dictionary of data frames, where the key of the dict is the column used as base of group by function in SQL query
        template -> 'column name' : data_frame
    """
    print(f'{time.ctime()}, Start downloading data from GBQ')
    # instantiates connector GBQ to a variable
    connector = ConnectorGBQ()
    print(f'{time.ctime()}, GBQ connector instantiated')
    # get data of canceled deliveries
    df_canc = connector.read_query(c.q_canc)
    print(f'{time.ctime()}, Canceled deliveries read')
    # get data of late deliveries
    df_late = connector.read_query(c.q_late)
    print(f'{time.ctime()}, Late deliveries read')
    # get data of on time deliveries
    df_on_time = connector.read_query(c.q_on_time)
    print(f'{time.ctime()}, On time deliveries read')
    # get groupped data frames
    dict_dfs = get_groupped_dfs()
    print(f'{time.ctime()}, All data was downloaded GBQ')
    return df_canc, df_late, df_on_time, dict_dfs
    
def get_data_predict():
    """
    Download from Google Big Query the needed data to predict new entries
    
    Parameters
    ----------
    None
    
    Results
    -------
    df_open : pandas.DataFrame
        Data frame with all open supplier deliveries
    dict_dfs : dict
        Dictionary of data frames, where the key of the dict is the column used as base of group by function in SQL query
        template -> 'column name' : data_frame
    """
    print(f'{time.ctime()}, Start downloading data from GBQ')
    # instantiates connector GBQ to a variable
    connector = ConnectorGBQ()
    # get data of open deliveries
    df_open = connector.read_query(c.q_open)
    print(f'{time.ctime()}, Open deliveries read')
    # get groupped data frames
    dict_dfs = get_groupped_dfs()
    print(f'{time.ctime()}, All data was downloaded GBQ')
    return df_open, dict_dfs

def format_dataframes_update(df_canc, df_late, df_on_time):
    """
    Format data frames used in the update/retrain model process
    
    Parameters
    ----------
    df_canc : pandas.DataFrame
        Data frame with canceled deliveries (status 90 and 91)
    df_late : pandas.DataFrame
        Data frame with late deliveries (status 40 and 60), and PSTNG_DT (receiving date) > SCHDL_PLAN_DVSN_DCMNT_DT (promissed delivery date)
    df_on_time : pandas.DataFrame
        Data frame with late deliveries (status 40 and 60), and PSTNG_DT (receiving date) <= SCHDL_PLAN_DVSN_DCMNT_DT (promissed delivery date)
    
    Results
    -------
    df_canc : pandas.DataFrame
        Data frame formatted
    df_late : pandas.DataFrame
        Data frame formatted
    df_on_time : pandas.DataFrame
        Data frame formatted
    """
    # for each data frame, set the columns PRCHS_NR and PRCHS_DCMNT_ITEM as index
    for df in [df_canc, df_late, df_on_time]:
        df.set_index(keys=['PRCHS_NR','PRCHS_DCMNT_ITEM'], inplace=True)
    print(f'{time.ctime()}, Data frames indexed')
    # set the target column for classificaion analysis
    # we want to know if a delivery is going to be late or canceled
    df_canc['TRGT'] = 1
    df_late['TRGT'] = 1
    # and, for exclusion, what have arrived on time
    df_on_time['TRGT'] = 0
    print(f'{time.ctime()}, Target columns set')
    return df_canc, df_late, df_on_time

def preprocessing_entries(df, dict_dfs):
    """
    Execute the whole preprocessing process, which includes adding features, scaling and one hot encoding
    
    Parameters
    ----------
    df_issue : pandas.DataFrame
        Data frame with the problematic deliveries
    df_on_time : pandas.DataFrame
        Data frame with deliveries which everything went well
    dict_dfs : list
        Dictionary with groupped data from categorical columns
        Follows this template -> {'categorical_column_name': 'df_with_groupped_data',}
    
    Results
    -------
    df : pandas.DataFrame
        Data frame after all treatments done to it
    """
    df = prep_data(df, dict_dfs)
    # apply a scaler to the numerical columns of the data frame
    df = scale_data_frame(df, scl_typ='StandardScaler') # ta aqui
    # apply one hot encoding to the categorical columns of the data frame
    df = one_hot_encoder(df, c.lst_cat_cols) # ta aqui
    return df

def save_model(df, model_name, save_mode, disp_scores):
    """
    Saves the trained model at defined folder
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data frame all data after the preprocessing
    model_name : string
        Name of the model, could be 'canc_deliv_gnb' or 'late_deliv_gnb'
    save_mode : string
        Saving mode, could be 'local' or 'DAG'
    disp_scores : boolean
        If it is True, besides the fact of training the model it displays the confusion matrix, AUC and the acurracy array of the model
        If it is not, simply trains the model.
        
    Results
    -------
    None
    """
    # set the features data frame and the target data frame
    X = df[df.drop('TRGT', axis=1).columns.tolist()]
    y = df[['TRGT']]
    print(f'{time.ctime()}, Feature and Target data frames splitted')
    print(f'{time.ctime()}, Start PCA tranformation')
    x_pca = pca_transform(c.pca_n_comps, X)
    print(f'{time.ctime()}, Start PCA transformation')
    # instantiates a Gaussian Naive Bayes model
    gnb = GaussianNB(var_smoothing=0.1)
    print(f'{time.ctime()}, Start fitting model')
    gnb.fit(x_pca, y)
    print(f'{time.ctime()}, Model fitted')
    # save the model to disk
    print(f'{time.ctime()}, Start saving model')
    pickle.dump(gnb, open(get_output_path(model_name+'.sav', save_mode, model_name), 'wb'))
    if disp_scores:
        # split it into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            pca_transform(c.pca_n_comps, X), y, test_size=0.8, random_state=42, shuffle=True)
        print(f'{time.ctime()}, Start fitting model')
        gnb.fit(X_train, y_train)
        print(f'{time.ctime()}, Model fitted')
        y_pred_gnb = gnb.predict(X_test)
        scores = cross_val_score(gnb, X, y, cv=5)
        print(f'{time.ctime()}, Prediction made')
        classifier_accuracy(y_test, y_pred_gnb)
        print(scores)
    print(f'{time.ctime()}, Model saved locally')

def pca_transform(n_components, X):
    """
    Apply the principle components analysis to the data frame and reshape the input data frame to the only needed columns
    
    Parameters
    ----------
    X : pandas.DataFrame
        Data frame with all data, some times it could be just a slice of the entire dataframe (split_train method)
    n_components : integer
        Number of princple components used in the PCA
    
    Results
    -------
    x_pca : array
        X before PCA transformation
    """
    pca = PCA(n_components)
    pca.fit(X)
    x_pca = pca.transform(X)
    return x_pca

def predict_new_entries(df_open, dict_dfs, model_name):
    """
    Predict the chance of new entries be of the target classification and save the result to the database
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with all data
    dict_dfs : list
        Dictionary with groupped data from categorical columns
        Follows this template -> {'categorical_column_name': 'df_with_groupped_data',}
    model_name : str
        Name of the model, could be 'canc_deliv_gnb' or TDB
    
    Results
    -------
    None
    """
    # how the preprocessing_entries needs a specific type of index set it to the data frame
    df_open = df_open.set_index(keys=['PRCHS_NR','PRCHS_DCMNT_ITEM'])
    # apply pre processing to the data frame
    df_f = preprocessing_entries(df_open, dict_dfs)
#     # apply pca using the n_components defined using the pca_optimal_n_components method
#     pca = PCA(n_components=c.pca_n_comps)
#     pca.fit(df_f)
#     x_pca = pca.transform(df_f)
    x_pca = pca_transform(c.pca_n_comps, df_f)
    print(f'{time.ctime()}, Start loading classification model')
    # load the model from disk
    gnb = pickle.load(open(get_output_path(model_name+'.sav', 'DAG', model_name), 'rb'))
    print(f'{time.ctime()}, Classification model loaded')
    print(f'{time.ctime()}, Start prediction')
    # probabilistic predict new entries
    y_pred_gnb = gnb.predict_proba(x_pca) 
    print(f'{time.ctime()}, New entries pedricted')
    print(f'{time.ctime()}, Start formatting data frame to GBQ table template')
    # how we need the index columns in the final database table we need to reset the index
    df_open = df_open.reset_index()
    # how the prediction results  into a list of lists which have the chance of P and 1-P
    # but P could be 0 or 1 depending on the sort of the inputed data frame
    # get_proba_one methods garatees that the result is going to be always for 1s %
    if model_name == 'canc_deliv_gnb':
        col_nm = 'CANC_PROBLTY'
        table_name = 'FT_CHNC_DLVRY_ISSU_CANC_PROBLTY'
        table_schema = c.schema_canc
        tec_user_updt = 'DAG_chnc_dlvry_issu_canc_problty_pred'
    if model_name == 'late_deliv_gnb':
        col_nm = 'LATE_PROBLTY'
        table_name = 'FT_CHNC_DLVRY_ISSU_LATE_PROBLTY'
        table_schema = c.schema_late
        tec_user_updt = 'DAG_chnc_dlvry_issu_late_problty_pred'
    df_open[col_nm] = get_proba_one(gnb, y_pred_gnb)
    # add a reference date
    df_open['REF_DT'] = datetime.now().date()
    # insert inpsection columns
    df_open['TEC_CRE_DT'] = datetime.now()
    df_open['TEC_UPT_DT'] = datetime.now()
    df_open['TEC_USER_UPT'] = tec_user_updt
    # reorder the columns
    df_open = df_open[['REF_DT', 'PRCHS_NR','PRCHS_DCMNT_ITEM', col_nm, 'TEC_CRE_DT', 'TEC_UPT_DT', 'TEC_USER_UPT']]
    print(f'{time.ctime()}, Columns added and reordered')
    # instantiates a GBQ connector
    connector = ConnectorGBQ()
    # upload the table to the database
    connector.export_table(
        df_open, 
        dataset=c.dataset, 
        table_name=table_name, 
        if_exists='append',
        table_schema=table_schema)
    print(f'{time.ctime()}, New entries prediction saved in the database')

def update_model(df_issue, df_on_time, dict_dfs, model_name, save_mode, disp_scores):
    """
    Preprocess the data, train the model and save it
    
    Parameters
    ----------
    df_issue : pandas.DataFrame
        Data frame with the problematic deliveries
    df_on_time : pandas.DataFrame
        Data frame with deliveries which everything went well
    dict_dfs : list
        Dictionary with groupped data from categorical columns
        Follows this template -> {'categorical_column_name': 'df_with_groupped_data',}
    model_name : string
        Name of the model, could be 'canc_deliv_gnb' or 'late_deliv_gnb'
    save_mode : string
        Saving mode, could be 'local' or 'DAG'
    disp_scores : boolean
        If it is True, besides the fact of training the model it displays the confusion matrix, AUC and the acurracy array of the model
        If it is not, simply trains the model.
        
    Results
    -------
    None
    """
    df_issue['VEND_NR'] = df_issue['VEND_NR'].apply(lambda x: int(x))
    # concatenate the issue dataframe with the on time data frame
    df = pd.concat([df_issue, df_on_time])
    print(f'{time.ctime()}, Issue and on time data frames concatenated')
    df = preprocessing_entries(df, dict_dfs)
    save_model(df, model_name, save_mode, disp_scores)
    
def get_proba_one(clf, y_pred):
    """
    How the prediction results into a list of lists which have the chance of P and 1-P, but P could be 0 or 1 depending on the sort of the inputed data frame
    This methods garatees that the result is always going to be for the 1s %
    
    Parameters
    ----------
    clf : object
        Classification method
    y_pred : list
        List of list, which I don't know if the P is for 0 or 1
    
    Results
    -------
    lst_proba : list
        List with only 1s chance
    """
    # clf.classes_ show the sequence used in the model, [0 1] or [1 0]
    # convert it to a list and use index to find the index of 1 value
    i = clf.classes_.tolist().index(1)
    # instantiates an empty list
    lst_proba = []
    for ele in y_pred:
        # for each element of the list save only the P %
        lst_proba.append(ele[i])
    return lst_proba

def train():
    """
    Retrain the models
    
    Parameters
    ----------
    None
    
    Results
    -------
    None
    """
    print(f'{time.ctime()}, Start of the process')
    df_canc, df_late, df_on_time, dict_dfs = get_data_train()
    df_canc, df_late, df_on_time = format_dataframes_update(df_canc, df_late, df_on_time)
    update_model(df_canc, df_on_time, dict_dfs, 'canc_deliv_gnb', c.save_mode, c.disp_scores)
    update_model(df_late, df_on_time, dict_dfs, 'late_deliv_gnb', c.save_mode, c.disp_scores)
    print(f'{time.ctime()}, End of the process')
    
def predict():
    """
    Predict new entries on the cancelation model
    
    Parameters
    ----------
    None
    
    Results
    -------
    None
    """
    print(f'{time.ctime()}, Start of the process')
    df_open, dict_dfs = get_data_predict()
    predict_new_entries(df_open, dict_dfs, model_name='canc_deliv_gnb')
    predict_new_entries(df_open, dict_dfs, model_name='late_deliv_gnb')
    print(f'{time.ctime()}, End of the process')    
   