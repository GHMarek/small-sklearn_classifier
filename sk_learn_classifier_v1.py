# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:14:42 2021

@author: marek.anuszewski
"""


'''
skrypt pisany pod typowanie spraw do egzekucji

skrypt ma zwracać predykcje wykonane przez wskazane klasyfikatory z biblioteki sklearn

~ workflow ~

1. import potrzebnych bibilotek, obiektów
2. import seta
3. przygotowanie funkcji potrzebnych do zbudowania klasyfikatora
4. przygotowanie zmiennych z wywoływanych funkcji
5. dodanie predykcji klasyfikatorów do testowanego seta
6. walidacja wybranych klasyfikatorów
7. zapis typowań do excela
8. testy - skrypt ułatwiający testowanie pojedynczych klasyfikatorów i ich walidację

'''
#%% pełen import
# to trzeba puścić pierwsze
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format # wyswietla floaty w formacie z dwoma miejsca po przecinku
pd.options.mode.chained_assignment = None # usuwa z konsoli wyswietlanie ostrzeżeń z pandas
    
    
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# chciałem odfiltrować częsc warnings, ale nie wiem czy to dziala
# no i czesc jednak lepiej zobaczyc
import warnings
import sklearn.exceptions
warnings.filterwarnings(action='ignore')
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

from random import randint
import time
from sys import stdout

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

from datetime import datetime #, timedelta #, date
datetimetoday = str(datetime.today().strftime('%m-%d-%Y-%H-%M-%S'))

input_file_path = r'D:\moje_vx\model_wyceny\egzekucja\\'
output_file_path = r'D:\moje_vx\model_wyceny\egzekucja\\'


#%% import seta
def f_import_set():
    '''
    zwraca zmienną df (pd.DataFrame) zawierającą set do analizy
    '''
    
    
    import pandas as pd
    # opcje wyswietlania tabel i wartosci w konsoli
    #pd.set_option('display.max_columns', None) # pokazuje wszystkie kolumny w konsoli, przy wyswietlaniu
    pd.options.display.float_format = '{:.2f}'.format # wyswietla floaty w formacie z dwoma miejsca po przecinku
    pd.options.mode.chained_assignment = None # usuwa z konsoli wyswietlanie ostrzeżeń z pandas
    
    # sprawdzenie wersji sklearn
    # import sklearn
    # sklearn.__version__
    # Out[35]: '0.23.1'
    

    # wczytanie danych z excela do zmiennej o nazwie "df" od pandas.DataFrame
    df = pd.read_excel(input_file_path + 'egzek1.xlsx', sheet_name = 'dane').reset_index()
    # df_backup = df.copy()
    return df

#%% funkcje
# zbiór funkcji z podstawowymi zmiennymi

# zmienne z odfiltrowanymi setami pasującymi do X, y
def f_df_filtr():
    
    '''
    zwraca dataframe'y, które powinny zawierać oryginalny set (prawdziwe kolumny z excela),
    podzielone tak jak X i X_test
    
    cel jest taki, żeby na koniec przyłączyć do niego wyniki predykcji
    '''
    
    return df.loc[df.Czybyłaegzekucja == 0], df.loc[df.Czybyłaegzekucja > 0] # bo klasa zero # bo klasa jeden

# przygotowanie seta w pandas - obróbka
def f_przygotowanie_seta():
    
    '''
    zwraca zmienne X, y, X_test
    
    te zmienne będą argumentami w metodach typu fit(), transform(), predict()
    
    tutaj należy modyfikować zakres kolumn, podział seta, wszelką obróbkę w pandas
    '''
    
    try:
        from pandas.tseries.offsets import MonthEnd
        # zmieniam daty na koniec miesiaca
        df.uko_data_cesji = pd.to_datetime(df.uko_data_cesji, format="mm-dd-yyyy") + MonthEnd(0)
        # zmieniam daty na liczbę porządkową daty względem 01-01-01 
        df.uko_data_cesji = df.uko_data_cesji.apply(lambda x: x.toordinal())
    except:
        pass
    
    
    X = df.loc[df.Czybyłaegzekucja > 0][['dlug','etap','uko_data_cesji','PierwSaldoEgze','obecny_dlug','WpłatyEtappolubowny'
                                     ,'Ilośćwpłat(polub)','WpłatyEtapsądowy','Ilośćwpłat(sąd)','AllKD',
                                     'AllObw','AllUgod','Rokurodzenia','Płeć','Typportfela','zbywca','DPD']]
    
    y = df.loc[df.Czybyłaegzekucja > 0]['wynikowa']
    
    # set do testowania
    X_test = df.loc[df.Czybyłaegzekucja == 0][['dlug','etap','uko_data_cesji','PierwSaldoEgze','obecny_dlug','WpłatyEtappolubowny'
                                     ,'Ilośćwpłat(polub)','WpłatyEtapsądowy','Ilośćwpłat(sąd)','AllKD',
                                     'AllObw','AllUgod','Rokurodzenia','Płeć','Typportfela','zbywca','DPD']]
    
    
    # df.isna().sum() # brak nan

    
    return X, y, X_test

# preprocessing
def f_preprocessing():
    '''
    zwraca ogólny (kodujący wszystkie kolumny) ColumnTransformer do wrzucenia w pipeline
    
    wykorzystane transformery:
    
    StandardScaler()
    OneHotEncoder(handle_unknown = 'ignore'
    '''

    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
    #from sklearn.impute import SimpleImputer
    
    # weryfikacja selectorów
    
    # df.select_dtypes(include = np.number).columns # continous
    # df.select_dtypes(include = object).columns # feature
    
    number_selector = make_column_selector(dtype_include=np.number) # select wszystkie z numerami
    object_selector = make_column_selector(dtype_include=object) # select wszystkie kolumny z obiektami
    # all_selector = make_column_selector() # tutaj chciałbym, żeby wybierał wszystkie
    
    my_scaler_column_trans = make_column_transformer( (StandardScaler(), number_selector), remainder='passthrough')
    my_ohe_column_trans = make_column_transformer( (OneHotEncoder(handle_unknown = 'ignore'), object_selector), remainder='passthrough')
    
    # imputer uzupełnia nan jesli dane są tylko typu np.number
    # my_imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
    
    # w zmiennej general_trans trzymam ColumnTransformer, który zawiera StandardScaler i OneHotEncoder do naszego setu
    # ten obiekt jest potem potrzebny do pipeline
    general_trans = ColumnTransformer(transformers=[("scaler", my_scaler_column_trans, number_selector), ("ohe", my_ohe_column_trans, object_selector)], remainder = 'passthrough')
    
    
    # general_trans.fit_transform(X) # tak sie sprawdza tablice przekształconą
    # my_scaler_column_trans.fit_transform(X)
    # my_ohe_column_trans.fit_transform(X)
    # jeżeli encoder ma metody get_feature_names to mozna je wyciagnac:
    # my_ohe_column_trans.get_feature_names() #
    
    # te feature_names() mozna polaczyc w koncu z pipeline0.steps[1][1].coef_
    # tylko pipeline0 musi oczywiscie istniec:

    # pd.DataFrame(zip(my_ohe_column_trans.get_feature_names(), pipeline0.steps[1][1].coef_[0]))
    

    
    return general_trans, my_ohe_column_trans

# estimators
def f_classifiers():
    '''
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    
    zwraca tylko klasyfikatory, w list[]:
    
    trzeba pilnować zgodnosci names z faktycznymi klasyfikatorami w liscie classifiers
    
    mozna zakomentowac, jesli za dlugo sie licza lub ich nie chcemy
    jesli cos usuwamy z listy to trzeba zmienic liste nazw (names)
    importy są nie po kolei więc lepiej zostawić
    
    przy wywołaniu wykonuje sie print z listy names
    
    
    '''
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    names = [
        "LogisticRegression"
            # , "KNeighborsClassifier"
            # , "Linear SVM"
            # , "RBF SVM"
            # , "Gaussian Process"
            # , "Decision Tree"
            # , "Random Forest"
            # , "Neural Net"
             , "AdaBoost"
             # , "Naive Bayes"
             # , "QDA"
             ]
    
    classifiers = [
    LogisticRegression(solver='lbfgs')
    #, KNeighborsClassifier(3)
    #, SVC(kernel="linear", C=0.025)
    #, SVC(gamma=2, C=1)
    #, GaussianProcessClassifier(1.0 * RBF(1.0))
    #, DecisionTreeClassifier(max_depth=5)
    #, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    #, MLPClassifier(alpha=1, max_iter=1000)
    , AdaBoostClassifier()
    #, GaussianNB()
    #, QuadraticDiscriminantAnalysis()
            ]
    
    print([x for x in names])

    return classifiers, names

# klasyfikator run
def f_klasyfikator_run():
    '''
    zwraca DataFrame z dodanymi kolumnami z predykcjami
    '''
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    
    import warnings
    import sklearn.exceptions
    warnings.filterwarnings(action='ignore')
    import time
    
    
    c = 0
    for i, j in zip(list_classifiers, c_names): # w kazdej iteracji bierze do zmiennych i, j klasyfikator i nazwe (string) z listy
        total_start = time.time() # zachowuje czas iteracji
        print('iteracja: ' + str(c) + '; klasyfikator: ' + str(j)) # print ktora iteracja, jaki klasyfikator (z c_names w zmiennej j)
        #########################
        
        pipeline0 = make_pipeline(general_trans, i) # robi pipeline z klasyfikatorem z i
        pipeline0.fit(X, y) # bierze zmienne z zewnatrz pętli, zawsze te same i jak rozumiem się nie zmieniają
        
        
        df_0['cls_' + str(j)] = pipeline0.predict(X_test) # tworzy kolumnę z nazwy klasyfikatora i przedrostka 'cls_', przypisuje do niej array z predictem klasyfikatora w tej iteracji pętli
        
        #########################
        print('iteracja: ' + str(c) + ' : ok; df_0 nowa kolumna: ' + df_0.columns[-1])    
        print('Elapsed time: ' + str(round((time.time() - total_start)/60,2))) # print czasu kończącej się iteracji
        c = c + 1 # zwiększa c, która jest potrzebna tylko do print   
    
    # za kazdym razem ta petla dorzuca kolumny do df_0, lepiej sobie nie zrobic smietnika
    # df_0, df_1 = f_df_filtr()
    
    return df_0

# klasyfikator walidacja

# tutaj jest pełna lista scoring parameters, których mozna uzywac do cross_val_score i cross_validation
# trzeba zwrórcić uwagę, jakie zmienne wchodzą do walidacji, i czy w odpowiednie kolejnosci y_true, y_pred
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

def f_walidacja_test():
    
    '''
    funkcja zwracająca kilka walidacji z sklearn
    
    printuje dane do konsoli i pliku txt w output_file_path

    '''
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import cross_validate
    from random import randint
    
    # plik do zapisania walidacji w txt
    from sys import stdout
    datetimetoday = str(datetime.today().strftime('%m-%d-%Y-%H-%M-%S'))
    f = open(output_file_path + 'test_' + datetimetoday + '.txt', 'w') # nazwa pliku do zapisu testu
    
    # trzeba sie w ogole zastanowic, gdzie tutaj maja wejsc dane setu testowego i na ktorych ma byc fit 
    from sklearn.model_selection import train_test_split
    split_X_train, split_X_test, split_y_train, split_y_test = train_test_split( X, y, test_size=0.33, random_state=randint(1, 42), shuffle = True)
    
    cc = 0
    for i, j in zip(list_classifiers, c_names): # w kazdej iteracji bierze do zmiennych i, j klasyfikator i nazwe (string) z listy
        total_start = time.time() # zachowuje czas iteracji
        if cc == 0: # w pierwszej iteracji zwraca nagłówek
            print('walidacja klasyfikatorów z funkcji f_classifiers()' + '\n'), print('walidacja klasyfikatorów z funkcji f_classifiers()' + '\n', file = f)
            print('rozmiar seta: ' + str(X.shape), '\n')
            print('wykorzystane kolumny:', '\n'), print('wykorzystane kolumny:', '\n', file = f)
            print(pd.Series(X.columns), '\n'), print(pd.Series(X.columns), '\n', file = f)
            print('wykorzystane klasyfikatory:' + '\n'), print('wykorzystane klasyfikatory:' + '\n', file = f)
            print(pd.Series(c_names), '\n'), print(pd.Series(c_names), '\n', file = f)
            print('ohe wspolczynniki cech: ' + '\n'), print('ohe wspolczynniki cech: ' + '\n', file = f)
            print(feat_importance, '\n'), print(feat_importance, '\n', file = f)
            
        print('*****************' + str(j) + '*****************' + '\n'), print('*****************' + str(j) + '*****************' + '\n', file = f)
        print('iteracja: ' + str(cc) + '; klasyfikator: ' + str(j) + '\n', end='\n') # print ktora iteracja, jaki klasyfikator (z c_names w zmiennej j)
        print('iteracja: ' + str(cc) + '; klasyfikator: ' + str(j) + '\n', file = f) 
        #########################
        
        pipeline0 = make_pipeline(general_trans, i) # robi pipeline z klasyfikatorem z i
        pipeline0.fit(split_X_train, split_y_train) # bierze zmienne z zewnatrz pętli, zawsze te same i jak rozumiem się nie zmieniają   
        
        y_pred = pipeline0.predict(split_X_test) # X czy X_test?
        y_true = split_y_test # prawdziwe odpowiedzi    
        
        #########################
        # 1
        k = 'sklearn.metrics - ' + 'accuracy_score'
        t = accuracy_score(y_true, y_pred, normalize=True)
        
        print(k), print(k, file = f)
        print(t , '\n', end='\n'), print(t, '\n', file = f)
           
        # 2
        k = 'sklearn.model_selection - ' + 'cross_val_score'
        t = pd.DataFrame(cross_val_score(pipeline0, X, y, cv=5, scoring='accuracy'))   
        
        print(k), print(k, file = f)
        print(t , '\n', end='\n'), print(t, '\n', file = f)
        
        # 3
        k = 'sklearn.metrics - ' + 'mean_absolute_error'
        mean_absolute_error(y_true, y_pred)
        t = accuracy_score(y_true, y_pred, normalize=True)      
        
        print(k), print(k, file = f)
        print(t , '\n', end='\n'), print(t, '\n', file = f)
        
        # 4    
        k = 'sklearn.metrics - ' + 'classification_report'
        t = classification_report(y_true, y_pred)    
        
        print(k), print(k, file = f)
        print(t , '\n', end='\n'), print(t, '\n', file = f)
        
        # 5
        k = 'sklearn.metrics - ' + 'confusion_matrix'
        t = pd.DataFrame(confusion_matrix(y_true, y_pred))  
       
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
        
        print(k), print(k, file = f)
        print(t , '\n', end='\n'), print(t, '\n', file = f)    
        
        
        #########################
        print('iteracja: ' + str(cc) + ' : ok;')    
        print('Elapsed time: ' + str(round((time.time() - total_start)/60,2)), '\n', end='\n') # print czasu kończącej się iteracji
        cc = cc + 1 # zwiększa cc, która jest potrzebna tylko do print   
    
    f.close()
    
    
    
    return None

def my_feature_importance():
    
    '''
    ma zwracac wagę cech, głównie z ohe, w pipeline0
    '''
    
    from sklearn.linear_model import LogisticRegression
    
    clff = LogisticRegression()
    pipeline0 = make_pipeline(general_trans, clff)
    pipeline0.fit(X,y)
    my_ohe_column_trans.fit_transform(X)
    feat_importance = pd.DataFrame(zip(my_ohe_column_trans.get_feature_names(), pipeline0.steps[1][1].coef_[0]))
    feat_importance[0] = feat_importance[0].str.replace('onehotencoder__','')
    
    feat_importance.plot(kind='barh', x = 0, figsize= (5,10), title = 'feature importances').get_figure()
    
    # feat_importance.plot(kind='barh', x = 0, figsize= (5,10), title = 'feature importances').get_figure().savefig(output_file_path + 'wykres_' + datetimetoday + 'pdf', dpi=480)
    
    return feat_importance

#%% klasyfikator przygotowanie
# po kolei wywołujemy kolejne funkcje, których returny przypisują się do zmiennych

# trzeba wczytać set (pierwszy krok)
df = f_import_set()

# df_0, df_1 to zmienne z setem z klasaim 0 i z setem z klasami 1, nieograniczone z pełbymi kolumnami
# do tych zmiennych potem zamierzam przyczepic wyniki klasyfikatorów

df_0, df_1 = f_df_filtr()


# teraz kolejne funkcje:
# pobieram przerobiony set
# help(f_przygotowanie_seta())
X, y, X_test = f_przygotowanie_seta()
#
# pobieram transformery
# help(f_preprocessing())
general_trans, my_ohe_column_trans = f_preprocessing()
#
# pobieram liste klasyfikatorów
# help(f_classifiers)
list_classifiers, c_names = f_classifiers()
#
feat_importance = my_feature_importance() # zapisuje feat_importance do zmiennej, jest potrzebne w walidacji
my_feature_importance() # zwraca feat_Importance + wykres, wywołanie nie jest potrzebne do kolejnych funkcji
#
# wykonuje prediction() z klasyfikatorami
# help(f_klasyfikator_run)
df_0 = f_klasyfikator_run()
#
# dostępne walidacje, print do konsoli i pliku
# help(walidacja_test)
f_walidacja_test()
#



feat_importance

#%% excel
total_start = time.time()
#########################
writer1 = pd.ExcelWriter(r'D:\moje_vx\model_wyceny\egzekucja\\' + 'wynik_' + datetimetoday + '.xlsx', engine='xlsxwriter', datetime_format = 'yyyy-mm-dd')

df_0.to_excel(writer1, sheet_name='df_0', index=False, startrow=0)

writer1.save()
#########################
print('Elapsed time: ' + str(round((time.time() - total_start)/60,2)))

#%% testy

# można używać tych fragmentów do wywoływania instancji pojedynczych klasyfikatorów
# w funkcji f_classifiers() powstaje lista klasyfikatorów i nazw
# żeby użyć jednego z nich należy wywołać je indeksem, według kolejnoci
# np. pierwszy klasyfikator z classifiers - classifiers[0], pierwsza nazwa names[0]

# df = f_import_set()
# df_0, df_1 = f_df_filtr()
# X, y, X_test = f_przygotowanie_seta()
# general_trans = f_preprocessing()
# list_classifiers, c_names = f_classifiers()

################################################################ klasyfikator
# test jednego wybranego klasyfikatora z list_classifiers
# from sklearn.pipeline import make_pipeline
# import time

# total_start = time.time()

# pipeline0 = make_pipeline(general_trans, list_classifiers[0]) # podaje index klasyfikatora do testu
# pipeline0.fit(X, y)
# pipeline0.predict(X_test)

# print('Elapsed time: ' + str(round((time.time() - total_start)/60,2)))
# # print nazw i klasyfikatorow, mozna sprawdzic czy sie zgadzaja
# c = 0
# for i, j in zip(c_names, list_classifiers):
#     print(c, i, j)
################################################################ klasyfikator
#
# ################################################################ walidacja
# # test walidacji jednego wybranego klasyfikatora z list_classifiers
# from sklearn.pipeline import make_pipeline
# import time

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_validate


# pipeline0 = make_pipeline(general_trans, list_classifiers[1]) # podaje index klasyfikatora do testu
# pipeline0.fit(X, y)

# y_pred = pipeline0.predict(X) # X czy X_test?
# y_true = y # prawdziwe odpowiedzi    

# # 1
# accuracy_score(y_true, y_pred, normalize=True)

# # 2
# cross_val_score(pipeline0, X, y, cv=5, scoring='accuracy').mean()

# # 3
# mean_absolute_error(y_true, y_pred)
# accuracy_score(y_true, y_pred, normalize=True)

# # 4

# print( classification_report(y_true, y_pred) ) #

# # 5

# pd.DataFrame(confusion_matrix(y_true, y_pred))


# ################################################################ walidacja

#