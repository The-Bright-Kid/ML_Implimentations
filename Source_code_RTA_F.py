''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''   Project Name : Road Traffic Severity Classification
            Domain : Machine Learning | Classification
             Owner : Shivahari Gopalakrishnan
           Purpose : The purpose of this project is to Explore the hidden insights from Road Traffic Accidents
                     of the year 2017 - 2018
  Data Description : This data set is collected from Addis Ababa Sub-city police departments for master's
                     research work. The data set has been prepared from manual records of road traffic accidents
                     of the year 2017-20. All the sensitive information has been excluded during data encoding and
                     finally it has 32 features and 12316 instances of the accident
    Target Feature : Accident_severity
 Metric evaluation : f1_score   '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import dabl
import plotly.subplots
import pandas               as pd
import matplotlib.pyplot    as plt
import matplotlib           as mpl
import numpy                as np
import seaborn              as sns
import kaleido              as kaleido
import missingno            as msno
import xgboost              as xgb
import shap                 as shap
import plotly               as py
import plotly.graph_objects as go
import plotly.io            as pio
pio.renderers.default = "browser"
plotly.offline.init_notebook_mode()
pio.renderers.default = 'png'
sns.set_style('darkgrid')
from   mpl_toolkits.mplot3d    import Axes3D
from   sklearn.model_selection import KFold
from   sklearn.model_selection import GridSearchCV
from   xgboost                 import XGBClassifier
from   sklearn.model_selection import train_test_split
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.ensemble        import ExtraTreesClassifier
from   imblearn.over_sampling  import SMOTE
from   collections             import Counter
from   sklearn.metrics         import accuracy_score
from   sklearn.metrics         import precision_score
from   sklearn.metrics         import recall_score
from   sklearn.metrics         import f1_score
from   sklearn.metrics         import confusion_matrix
from   sklearn.metrics         import classification_report
from   sklearn                 import preprocessing
from   sklearn.preprocessing   import LabelEncoder
from   IPython.display         import display

rta    = pd.read_csv(dabl.datasets.data_path(r'F:\Shivahari_Projects\trimester 5\ML Comp-MGP\RTA_ML PROJECT1\RTA Dataset.csv'))

'''''''''''''''''''''Exploratory Data Analysis [ EDA ]'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
pd.set_option ('display.max_columns', None)
pd.set_option ('display.max_rows', None)
print ("The Total no. of Features and Instances are",  rta.shape)
print ("Taking a Quick Glance into how my data looks",  rta.head)
rta['Time'] = pd.to_datetime(rta['Time'])
rta_clean   = dabl.clean(rta,   verbose=0)
types       = dabl.detect_types(rta_clean)
print ("Quick Marker for my missing values & other dirty columns", types)
print ("Quick analysis for describing main features of numericals in my data",  rta.describe())
print ("\n", rta.info())
print ("Quick analysis for describing main features of categories in my data",  rta.describe(include='object'))
rta.hist(figsize=(8,8), xrot=45)
plt.show()

for col in rta.select_dtypes(include='object'):
    if rta[col].nunique() <= 22:
        sns.countplot(y=col, data=rta)
        plt.show()

label_encoder = preprocessing.LabelEncoder()
y    = label_encoder.fit_transform(rta['Accident_severity'])
for col in rta.select_dtypes(include='object'):
    if rta[col].nunique() <=4:
        display(pd.crosstab(y, rta[col], normalize='index'))

for col in rta.select_dtypes(include='object'):
    if rta[col].nunique() <= 4:
        g = sns.catplot(x = col, kind='count', col = 'Accident_severity', data=rta, sharey=False)
        g.set_xticklabels(rotation=60)
        plt.show()

corr = rta.corr()
print ("Looking into Correlation Matrix", corr)
plt.figure(figsize=(6,6))
sns.heatmap(corr, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
plt.show()

sns.violinplot(x="Number_of_casualties", y="Vehicle_movement", data=rta)
plt.show()
sns.violinplot(x="Number_of_casualties", y="Type_of_collision", data=rta, color="Yellow")
plt.show()

'''''''''''''''''''''''''DATA PREPROCESSING'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''DETECTING THE MISSING VALUES & VISUAL MAPPING'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def missing_values_table(rta):
    mis_val = rta.isnull().sum()
    print("\nThe total number of missing values are ", mis_val)
    mis_val_percent = 100 * rta.isnull().sum() / len(rta)
    print("\nThe percentage of missing values are ", mis_val_percent)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    print("\nTable of total missing values and its percentages", mis_val_table)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print("The master dataframe for Road Traffic Accidents has " + str(rta.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns
missing_values_table(rta)

msno.bar(rta)                                     #Identifying amount of data missing
plt.show()
msno.matrix(rta)                                  #Plotting patterns of data missing
plt.show()
msno.matrix(rta.sample(100))                      #Plotting 1st 100 rows (sub-strata)
plt.show()
sorted_wos = rta.sort_values('Work_of_casuality') #Identifying correlation of my missing datapoints
msno.matrix(sorted_wos)
plt.show()
sorted_syv = rta.sort_values('Service_year_of_vehicle')
msno.matrix(sorted_syv)
plt.show()
msno.heatmap(rta)
plt.show()
msno.dendrogram(rta)                              #Using Dendogram to figure more patterns
plt.show()
                                                  #Dropping columns with high level of imbalance
rta.drop(columns= ['Defect_of_vehicle', 'Vehicle_driver_relation', 'Work_of_casuality', 'Fitness_of_casuality',], inplace=True)
impute_cols = [x for x in rta.isna().sum()[rta.isna().sum()!=0].index.tolist()]
for feat in impute_cols:
    mode = rta[feat].mode()[0]
    rta[feat].fillna(mode, inplace=True)
print(rta.isna().sum())

''''''''''''''''''''''''''''''''''''''''FEATURE ENGINEERING & ENCODING RTA FEATURES'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def ordinal_encoder(rta, feats):
    for feat in feats:
        feat_val  = list (1+np.arange(rta[feat].nunique()))
        feat_key  = list (rta[feat].sort_values().unique())
        feat_dict = dict (zip(feat_key, feat_val))
        rta[feat] = rta[feat].map(feat_dict)
    return rta                                    #Visualizing the frame after Encoding
rta = ordinal_encoder(rta, rta.drop(['Accident_severity'], axis=1).columns)
print ("The shape of my RTA data after Ordinal Encoding", rta.head())
for col in rta.drop('Accident_severity', axis=1):
    g = sns.FacetGrid(rta, col= 'Accident_severity', aspect= 1.2, sharey= False)
    g.map(sns.countplot, col, palette= 'Dark2')
    plt.show()
plt.figure(figsize=(22, 17))                      #Plotting correlation map    
sns.set(font_scale= 0.8)
sns.heatmap(rta.corr(), annot= True, cmap= plt.cm.get_cmap("CMRmap_r"))
plt.show()
                                                  #Splitting RTA Data
X = rta.drop('Accident_severity', axis= 1)
Y = rta['Accident_severity']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 42)
print ("Quick view to train and test dataframes\n", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
le      = LabelEncoder()
Y_train = le.fit_transform(Y_train)
counter = Counter(Y_train)
print ("---------------------------------------------------------")
for k,v in counter.items():
    per = 100*v/len(Y_train)
    print (f"Class= {k}, n= {v}, ({per: .2f}%)")
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)
counter = Counter(Y_train)
print ("---------------------------------------------------------")
for k,v in counter.items():
    per = 100*v/len(Y_train)
    print (f"Class= {k}, n= {v}, ({per: .2f}%)")
print ("---------------------------------------------------------")
print ("The shape of up|over sampled RTA Dataset: ", X_train.shape, Y_train.shape)

'''''''''''''''''''''''''''''''''''''BASELINE MODELING'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Y_test = le.fit_transform(Y_test)
def modelling(X_train, Y_train, X_test, Y_test, **kwargs):
    scores = {}
    models = []
    if  'xgb' in kwargs.keys() and kwargs['xgb']:
        xgb     = XGBClassifier()
        xgb.fit(X_train._get_numeric_data(), np.ravel(Y_train, order= 'C'))
        global Y_pred
        Y_pred         = xgb.predict(X_test._get_numeric_data())
        scores['xgb']  = [accuracy_score(Y_test, Y_pred)]
    if 'rf'  in kwargs.keys() and kwargs['rf']:
        rf = RandomForestClassifier(n_estimators= 200)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_test)
        scores['rf'] = [accuracy_score(Y_test, Y_pred)]
        models.append(rf)
    if 'extree'  in kwargs.keys() and kwargs['extree']:
        extree = ExtraTreesClassifier()
        extree.fit(X_train, Y_train)
        Y_pred = extree.predict(X_test)
        scores['extree'] = [accuracy_score(Y_test, Y_pred)]
        models.append(extree)
    return scores
print ("My Baseline Model Output: \n")
print (modelling(X_train, Y_train, X_test, Y_test, xgb= True, rf= True, extree= True))
                                                  #Model Performance and Show Metrics
def model_performance(model, Y_test, Y_hat):
    conf_matrix = confusion_matrix(Y_test, Y_hat)
    trace1 = go.Heatmap(z= conf_matrix, x= ["0 (pred)", "1 (pred)", "2 (pred)"],
                        y= ["0 (true)", "1 (true)", "2 (true)"], xgap= 2, ygap= 2,
                        colorscale= 'viridis', showscale= False)
    Accuracy     = accuracy_score  (Y_test, Y_hat)
    Precision    = precision_score (Y_test, Y_pred, average= 'weighted')
    Recall       = recall_score    (Y_test, Y_pred, average= 'weighted')
    F1_score     = f1_score        (Y_test, Y_pred, average= 'weighted')
    show_metrics = pd.DataFrame (data= [[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors       = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2       = go.Bar(x= (show_metrics[0].values),
                          y= [Accuracy, Precision, Recall, F1_score], text= np.round_(show_metrics[0].values, 4),
                              textposition = 'auto',
                              orientation  = 'h'   , opacity= 0.8,
                              marker       = dict(color = colors ,
                              line         = dict(color= '#000000', width= 1.5)))
    model = model                 #Plots
    fig   = plotly.subplots.make_subplots(rows=2, cols=1, print_grid= False,
                                          subplot_titles= ('Confusion_Matrix', 'Metrics'))
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig['layout'].update(showlegend    = False, title= '<b>Model performance report</b><br>' + str(model),
                         autosize      = True, height= 800, width= 800,
                         plot_bgcolor  = 'rgba(240, 240, 240, 0.95)',
                         paper_bgcolor = 'rgba(240, 240, 240, 0.95)',
                        )
    fig.layout.titlefont.size = 14
    py.offline.iplot(fig)
    fig.show(renderer= "browser")
extree = ExtraTreesClassifier()
extree.fit(X_train, Y_train)
Y_pred = extree.predict(X_test)
print ("Hyper parameters | input options: \n", extree.get_params())
model_performance(extree, Y_test, Y_pred)

'''''''''''''''''''''''''''''''''''''''''HYPER PARAMETER TUNING'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
gkf = KFold(n_splits= 3, shuffle= True, random_state= 42).split(X= X_train, y= Y_train)
                                  #Parameter grid of ETrees
params = {
    'n_estimators'      : range(100, 500, 100),
    'ccp_alpha'         : [0.0, 0.1],
    'criterion'         : ['gini'],
    'max_depth'         : [5,11],
    'min_samples_split' : [2,3],
         }
extree_estimator = ExtraTreesClassifier()
g_search = GridSearchCV(
                         estimator = extree_estimator,
                        param_grid = params,
                           scoring = 'f1_weighted',
                            n_jobs = 1,
                                cv = gkf,
                           verbose = 3,
                       )
extree_model = g_search.fit(X= X_train, y= Y_train)
print ("Best Score post tuning: \n", g_search.best_params_, g_search.best_score_)

extree_tuned = ExtraTreesClassifier(
                                    ccp_alpha          = 0.0,
                                    criterion          = 'gini',
                                    min_samples_split  = 2,
                                    class_weight       = 'balanced',
                                    max_depth          = 15,
                                    n_estimators       = 400
                                   )
extree_tuned.fit(X_train, Y_train)
Y_pred_tuned = extree_tuned.predict(X_test)
print("Final tuned throughput: \n", Y_pred_tuned)
print(np.concatenate((Y_pred_tuned.reshape(len(Y_pred_tuned),1), Y_test.reshape(len(Y_test),1)),1))

'''''''''''''''''''''''''''''''''''''''EXPLAINABLE - AI [SHAP]'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
shap.initjs()
X_sample = X_train.sample(500)
print ("Sample taking for shap value calculation\n", X_sample)
shap_values = shap.TreeExplainer(extree_tuned).shap_values(X_sample)
print ("Printing the shap values of identified sample\n", shap_values)
print ("Shap Summary Plot : Below\n")
shap.summary_plot(shap_values, X_sample, plot_type="bar")
shap.summary_plot(shap_values, X_sample, max_display=28)
shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0],
                shap_values[0][:],
                X_sample)
print (Y_pred_tuned[50])
shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0], 
                                   shap_values[1][50], 
                                   X_sample.iloc[50])
i=13
print (Y_pred_tuned[i])
shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0], 
                                   shap_values[0][i], 
                                   X_sample.values[i], 
                                   feature_names = X_sample.columns)
print (Y_pred_tuned[10])
row = 10
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row],
                                     base_values=shap.TreeExplainer(extree_tuned).expected_value[0], 
                                     data=X_sample.iloc[row],
                                     feature_names=X_sample.columns.tolist()))
shap.dependence_plot('Day_of_week', shap_values[2], X_sample)
shap.dependence_plot('Age_band_of_driver', shap_values[2], X_sample)
print(Y_pred_tuned[10])
shap.decision_plot(shap.TreeExplainer(extree_tuned).expected_value[0],
                   shap_values[2][:10],
                   feature_names=X_sample.columns.tolist())