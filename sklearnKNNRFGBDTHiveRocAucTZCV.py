# --coding:utf-8--
import matplotlib
matplotlib.use('Agg')
import os
import sys
import pymysql as MySQLdb
import ConfigParser
import numpy as np
import scipy as sc
import pandas as pd
import time, random
from pyhive import hive
from sklearn import metrics
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import interp
from numpy import linspace
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import mysql.connector
import cx_Oracle

STDSCALER = None
HASH_VALUE = None
reload(sys)
sys.setdefaultencoding('utf-8')
# import importlib,sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

def featureExtract(filePath, savePath, paras):
    trainX, trainy, columns = getData_2(filePath)
    # columns包涵df的表头信息
    strArr = paras.split(',')
    limit = strArr[len(strArr) - 1]
    model = gradient_boosting_classifier(trainX, trainy, ','.join(strArr[:len(strArr) - 1]))
    sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (30, startT)
    mysqlProBar(sqlProgressBarUpdate)
    feartureArr = model.feature_importances_
    index = 0
    tmp = []

    for fnum in feartureArr:
        fnum4 = '%.4f' % float(fnum)
        if float(fnum4) > float(limit):
            tmp.append([index, fnum4, columns[index]])
        index += 1

    tmp = sorted(tmp, key=lambda t: t[1], reverse=True)

    # 生成重要度表格
    fstStr1 = ['<tr>']
    fnlstr1 = ['<tr>']
    for tup1 in tmp:
        fstStr1.append('<td>%s</td>' % tup1[2])
        fnlstr1.append('<td>%s</td>' % tup1[1])
    fstStr1.append('</tr>')
    fnlstr1.append('</tr>')

    # 生成相关系数表格
    fstStr2 = ['<tr><td></td>']
    fnlstr2 = []
    for tup1 in tmp:
        str1 = '<tr><td>%s</td>' % tup1[2]
        fstStr2.append('<td>%s</td>' % tup1[2])
        for tup2 in tmp:
            piersonNum = '%.4f' % sc.stats.pearsonr(trainX[:, tup1[0]], trainX[:, tup2[0]])[0]
            t2 = '<td style="color:red">%s</td>' % str(piersonNum)
            str1 += t2
        fnlstr2.append(str1 + '</tr>')

    fstStr2.append('</tr>')
    ph = os.path.split(savePath)
    ph2 = os.path.split(filePath)[1]
    phTail = ph2[:ph2.rfind('_')]
    global datetime
    datetime = ph[1][:14]
    global rstSavePath
    rstSavePath = ph[0] + '/' + datetime + '_' + 'FE_' + phTail + '_predict_rst.html'
    datetime =datetime
    output = open(rstSavePath, "w")

    info = '\n'.join(
        ['<table border="1">', ''.join(fstStr1), ''.join(fnlstr1), '</table>', '<br/>', '<table border="2">',
         ''.join(fstStr2), '\n'.join(fnlstr2), '</table>'])

    output.write(info)
    try:
        output.flush()
        output.close()
        del output
    except Exception as e:
        print ('TZ extract Exception as e:' + e.message)
    sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (100, startT)
    mysqlProBar(sqlProgressBarUpdate)
    print ('Features extract Finished!')
    exit(0)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def ToMysql(sql):
    confPsr = ConfigParser.ConfigParser()
    global args
    mysqlConfDir=os.path.split(args[0])[0]+'/mysqlConf_local.ini';
    confPsr.read(mysqlConfDir)
    hostName = confPsr.get('base_info', 'HOST')
    portN = confPsr.get('base_info', 'PORT')
    Uname = confPsr.get('base_info', 'USERNAME')
    passW = confPsr.get('base_info', 'PASSWORD')
    DBName = confPsr.get('base_info', 'DATABASE')
    try:
        conn = MySQLdb.connect(host=hostName, port=int(portN), user=Uname, passwd=passW, db=DBName)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()
        return rows
    except MySQLdb.Error,e:
        print ('Mysql Error %d : %s' % (e.args[0], e.args[1]))


def mysqlProBar(sql):
    confPsr = ConfigParser.ConfigParser()
    global args
    mysqlConfDir = os.path.split(args[0])[0] + '/mysqlConf_local.ini';
    confPsr.read(mysqlConfDir)
    hostName = confPsr.get('base_info', 'HOST')
    portN = confPsr.get('base_info', 'PORT')
    Uname = confPsr.get('base_info', 'USERNAME')
    passW = confPsr.get('base_info', 'PASSWORD')
    DBName = confPsr.get('base_info', 'DATABASE')
    timeOut = confPsr.get('base_info', 'CONNECT_TIMEOUT')
    autoCommit = confPsr.get('base_info', 'AUTOCOMMIT')
    try:
        global connection
        if not connection:
            connection = MySQLdb.connect(host=hostName, port=int(portN), user=Uname, passwd=passW, db=DBName,
                                         autocommit=autoCommit, connect_timeout=int(timeOut))
        cur = connection.cursor()
        cur.execute(sql)

        connection.commit()


    except MySQLdb.Error,e:
        print ('Mysql Error %d : %s' % (e.args[0], e.args[1]))


def precisionTest(fPath, label, paras ):
    if str(fPath).endswith('.csv'):
        df = pd.read_csv(fPath)
    elif str(fPath).endswith('.xlsx'):
        df = pd.read_excel(fPath, sep='\t')
    elif str(fPath).endswith('.sql'):
        flist = open(fPath).readlines()
        count = len(flist)
        if count != 6:
            print ('sql file require 5 lines,only support 1 sql qurey.')
            exit(1)
        hostN = flist[0].split('=')[1].strip()
        portN = flist[1].split('=')[1]
        dbType=flist[2].split('=')[1].strip()
        userN = flist[3].split('=')[1].strip()
        passWd = flist[4].split('=')[1].strip()
        sqlStr = flist[5].strip()
        conn=None
        print (hostN, portN, dbType, userN, passWd, sqlStr)
        if(dbType=="hive"):
            conn = hive.Connection(host=str(hostN), port=int(portN), username=str(userN), password=str(passWd),
                                   auth='LDAP',
                                   thrift_transport=None)
        elif(dbType=="mysql"):
            conn = mysql.connector.connect(host=str(hostN), port=int(portN), user=str(userN), password=str(passWd),
                                   charset='utf8')
        elif (dbType == "oracle"):
            pass
        df = pd.read_sql(sqlStr, conn)
        try:
            conn.close()
        except Exception as e:
            print (e.message)
            print ("conn not closed properly.")
            del conn
    a, b = df.shape
    print (a,b,'a,b.shape')
    df = df.fillna(value=0)
    i = 0
    dfy = pd.DataFrame()
    if df.iloc[:, i].dtype == np.dtype('O'):
        SeriesY = df.iloc[:, i].apply(hash)

        dfy['rst'] = SeriesY

    else:
        dfy['rst'] = df.iloc[:, i]
    trainY = dfy.values
    print(trainY.shape,' :trainY.shape')
    dfx = pd.DataFrame()
    i += 1
    while i < b:
        if df.iloc[:, i].dtype == np.dtype('O'):
            SeriesX = df.iloc[:, i].apply(hash)
            dfx[str(i)] = SeriesX
            i += 1
        else:
            dfx[str(i)] = df.iloc[:, i]
            i += 1
    trainX = dfx.values
    global STDSCALER
    if (STDSCALER == None):
        STDSCALER = preprocessing.MinMaxScaler().fit(trainX)

    scaled_X = STDSCALER.transform(trainX)
    global trainSize
    try:
        trainSize=float(trainSize)#尝试将trainSize强转为float，转换失败则直接赋值为0.9
    except Exception as e:
        trainSize=0.9
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, trainY, train_size=trainSize, random_state=100)
    # 使用最优模型
    global model
    predictRst = model.predict(X_test)
    count = 0
    for pre, realy in zip(predictRst, y_test):
        if (pre == realy):
            count += 1
    accury = count * 1.0 / len(y_test)
    print ('My accury:%.2f%%' % (100 * accury))
    global kappaScore
    kappaScore = cohen_kappa_score(y_test, predictRst)
    print ("kappaScore:", kappaScore)

    cnf_matrix = confusion_matrix(y_test, predictRst)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    global cmf
    cmf = os.path.split(predictRstPath)[0] + '/' + os.path.split(predictRstPath)[1][:15] + label + "_CM.png"
    plt.savefig(cmf)
    accuracy = metrics.accuracy_score(y_test, predictRst)
    print ('accuracy: %.2f%%' % (100 * accuracy))
    return accuracy * 1.0000


def getMeanAucBestModel(label, X, y):
    print (X.shape, type(y))
    ytmp = [i[0] for i in y]
    y = np.array(ytmp)

    cv = StratifiedKFold(n_splits=10, shuffle=True) #十折运算
    totalAuc = 0.0
    outerAucV = 0.0
    outerAucDic = {}
    for i in range(1):
        innerAucV = 0.0
        innerAucDic = {}
        for trainIndex, testIndex in cv.split(X, y):
            global classifiers
            innerModel = classifiers[label](X[trainIndex], y[trainIndex],paras)
            innerPredict = innerModel.predict_proba(X[testIndex])
            fpr, tpr, thresholds = roc_curve(y[testIndex], innerPredict[:, 1])
            inner_auc = auc(fpr, tpr)
            totalAuc += inner_auc
            if inner_auc > innerAucV:
                innerAucV = inner_auc
            innerAucDic[inner_auc] = innerModel
        if innerAucV > outerAucV:
            outerAucV = innerAucV
        outerAucDic[innerAucV] = innerAucDic[innerAucV]
    bestModel = outerAucDic[outerAucV]
    meanAuc = totalAuc / 10*1
    return meanAuc, bestModel


def drawAUC(model, X, y):
    if model == None:
        print ('Error:No model')
        exit(1)

    #
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)#np.ndarry类型。
    start = 0.0
    stop = 1.0
    number_of_lines = 10
    cm_subsection = linspace(start, stop, number_of_lines)
    random.shuffle(cm_subsection)
    colors = [cm.jet(x) for x in cm_subsection]
    color=colors[int(random.random()*10)]
    lw = 2
    i = 0
    # k折交叉验证

    # ytmp = [i[0] for i in y]
    # y = np.array(ytmp)
    # for (trainIndex, testIndex), color in zip(cv.split(X, y), colors):
    #     probas_ = model.predict_proba(X[testIndex])
    #     fpr, tpr, thresholds = roc_curve(y[testIndex], probas_[:, 1])
    #     # print thresholds
    #     # print probas_[:,1]
    #     mean_tpr += interp(mean_fpr, fpr, tpr)
    #     mean_tpr[0] = 0.0
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, lw=lw, color=color,
    #              label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    #
    #     i += 1
    probas_ = model.predict_proba(X)
    fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])

    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='diagonal')
    mean_tpr /= 1
    # mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print ('mean_auc://', mean_auc)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='AUC (area = %0.2f)' % mean_auc, lw=lw)
    if meanAuc:
        plt.plot(mean_fpr, mean_tpr, color='r', linestyle='--',
             label='CLFMeanAUC (area = %0.2f)' % meanAuc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    global aucf
    aucf = os.path.split(predictRstPath)[0] + '/' + os.path.split(predictRstPath)[1][:15] + label + "_AUC.png"
    plt.savefig(aucf)
    plt.close()


def drawPR(model, X, y):
    if model == None:
        print ('Error:No model')
        exit(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_score = model.predict(X_test)

    average_precision = average_precision_score(y_test, y_score)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
#函数式画图，设定步长
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    global prSavePath
    prSavePath = os.path.split(predictRstPath)[0] + '/' + os.path.split(predictRstPath)[1][:15] + label + "_PR.png"
    plt.savefig(prSavePath)


def getData_2(fPath):
    if os.path.exists(fPath):
        df = None
        if str(fPath).endswith('.csv'):
            df = pd.read_csv(fPath,parse_dates=False)
        elif str(fPath).endswith('.xlsx'):
            df = pd.read_excel(fPath, sep='\t',parse_dates=False)
        elif str(fPath).endswith('.sql'):
            flist = open(fPath).readlines()
            count = len(flist)
            if count != 6:
                print ('sql file require 5 lines,only support 1 sql qurey.')
                exit(1)
            hostN = flist[0].split('=')[1].strip()
            portN = flist[1].split('=')[1]
            dbType = flist[2].split('=')[1].strip()
            userN = flist[3].split('=')[1].strip()
            passWd = flist[4].split('=')[1].strip()
            sqlStr = flist[5].strip()
            conn = None
            print (hostN, portN, dbType, userN, passWd, sqlStr)
            if (dbType == "hive"):
                conn = hive.Connection(host=str(hostN), port=int(portN), username=str(userN), password=str(passWd),
                                       auth='LDAP',
                                       thrift_transport=None)
            elif (dbType == "mysql"):
                conn = mysql.connector.connect(host=str(hostN), port=int(portN), user=str(userN), password=str(passWd),
                                               charset='utf8')
            elif (dbType == "oracle"):
                pass
                # conn = cx_Oracle.connect(str(userN), str(passWd), str(hostN) + ":" + portN + "/orcl")
            df = pd.read_sql(sqlStr, conn)
            try:
                conn.close()
                del conn
            except Exception as e:
                print (e.message,'Close Del Conn Exception as e!')
        a, b = df.shape
        df = df.fillna(value=0)
        i = 0
        dfy = pd.DataFrame()
        if df.iloc[:, i].dtype == np.dtype('O'):
            SeriesY = df.iloc[:, i].apply(hash)
            hashY_Y = zip(SeriesY, df.iloc[:, i])
            global HASH_VALUE
            HASH_VALUE = dict(hashY_Y)
            dfy['rst'] = SeriesY

        else:
            dfy['rst'] = df.iloc[:, i]
        trainY = dfy.values
        dfx = pd.DataFrame()
        i += 1
        while i < b:
            if df.iloc[:, i].dtype == np.dtype('O'):
                SeriesX = df.iloc[:, i].apply(hash)
                dfx[str(i)] = SeriesX
                i += 1
            else:
                dfx[str(i)] = df.iloc[:, i]
                i += 1
        trainX = dfx.values
        print (df.dtypes)
        global STDSCALER
        if (STDSCALER == None):
            STDSCALER = preprocessing.MinMaxScaler().fit(trainX)

        scaled_X = STDSCALER.transform(trainX)
        print (scaled_X.shape)
        return scaled_X, trainY, df.columns
    else:
        print ('No such file or directory!')


def writeResult(predict, path):
    if os.path.exists(path):
        os.remove(path)
    listy = []
    if HASH_VALUE:
        print (HASH_VALUE)
        for item in (pd.DataFrame(predict)).iloc[:, 0]:
            listy.append(HASH_VALUE[item])
        newSeriesY = pd.Series(listy)
        newdfY = pd.DataFrame(newSeriesY,columns=[testX.columns[0]])
        finalDFY = newdfY.join(testX.iloc[:, 1:])
    else:
        print ('****************')
        print  (testX.dtypes)
        print ('****************')
        finalDFY = pd.DataFrame(predict,columns=[testX.columns[0]]).join(pd.DataFrame(testX.iloc[:, 1:]))
    ph = os.path.split(path)
    global trainFPath
    ph2=os.path.split(trainFPath)[1]
    phTail=ph2[:ph2.rfind('_')]
    global datetime
    datetime = ph[1][:14]
    global rstSavePath
    rstSavePath = ph[0] + '/' + datetime + '_' + label + '_'+phTail+'_predict_rst.csv'
    datetime = int(datetime)
    finalDFY.to_csv(rstSavePath, sep=",", index=False)
    print ('Saved csv result!!')


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y, paras):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier

def knn_classifier(train_x, train_y, paras):
    from sklearn.neighbors import KNeighborsClassifier
    model = None
    #paras是x,x,x,x模式字符串，最少含有一个null
    parArr=paras.split(',')
    if parArr[0]!='null':
        model = KNeighborsClassifier(int(parArr[0]))
    else:
        model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y, paras):
    from sklearn.linear_model import LogisticRegression
    # paras是x,x,x,x模式字符串，最少含有一个null
    parArr = paras.split(',')
    if parArr[0] != 'null':
        #parArr参数待定
        model = LogisticRegression(penalty='l2')
    else:
        model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y, paras,):
    from sklearn.ensemble import RandomForestClassifier
    # paras是x,x,x,x模式字符串，最少含有一个null
    parsArr = paras.split(',')
    if len(parsArr) == 5:
        # parArr参数待分配,由于是字符串数字，需要转化对应的值
        #     print 'NEW ParaM'
        n_estimatorsV=parsArr[0]
        criterionV=parsArr[1]
        max_depthV=parsArr[2]
        minSamplesleafV=parsArr[3]
        max_featuresV = parsArr[4]
        # 讲none str转化为None空值。
        def str2None(*args):
            for item in args:
                try:
                    item+''  #类型检查
                    if item.lower() != 'none':
                        item = int(item)
                    else:
                        item = None
                except TypeError:
                    item=None
                    print (e.message)
                finally:
                    yield item



        max_depthV=str2None(max_depthV).next()#这个值只有none和整数

        # 如果最小分裂点样本需求数目、叶节点最小样本数、最佳分裂点最大特征数。接收int、float值。对0-1的float值，float话，其它值int化
        def init2Int(*args):
            for item in args:
                try:
                    item = float(item)
                    if 0 < item < 1.0:
                        pass
                    else:
                        raise TypeError
                except TypeError:
                    item = int(item)
                    print (e.message)
                finally:
                    yield item

        # a,b=[1,2] ->a=1 b=2
        minSamplesleafV=init2Int(minSamplesleafV).next()

        # 如果检验方法不是gini和entropy则初始为gini

        try:
            n_estimatorsV = int(n_estimatorsV)
        except TypeError:
            n_estimatorsV=10
        if criterionV.lower() not in ['gini','entropy']:
            criterionV='gini'
        if max_featuresV.lower()!='auto':
            try:
                if 0<abs(float(max_featuresV))<=1:
                    max_featuresV=float(max_featuresV)
                else:
                    max_featuresV=int(max_featuresV)
            except TypeError:
                max_featuresV='auto'
                print (e.message)

        model = RandomForestClassifier(criterion=criterionV,
                max_depth=max_depthV, max_features=max_featuresV,
                min_samples_leaf=minSamplesleafV,
                n_estimators=n_estimatorsV)
    else:
        model = RandomForestClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    return model
# RFCV Classifier using cross validation
def random_forest_cross_validation(train_x, train_y, paras):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import GradientBoostingClassifier
    parsArr = paras.split(',')
    for index,item in enumerate(parsArr):
        parsArr[index]=item.replace(';',',').split(',')
        for subindex,subitem in enumerate(parsArr[index]):
            if subitem.lower()=='none':
                parsArr[index][subindex]=None
            else:
                try:
                    parsArr[index][subindex] = int(subitem)
                    if parsArr[index][subindex]==0:
                        raise Exception('Error,it is a float')
                except Exception as e:
                    try:
                        parsArr[index][subindex] = float(subitem)
                    except Exception as e:
                        parsArr[index][subindex] = subitem
                        pass


    parNameArr=['n_estimators','criterion','max_depth','min_samples_leaf','max_features']
    ListParaV=zip(parNameArr,parsArr)
    dictParaV=dict(ListParaV)
    print (dictParaV)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=42)
    # cv = KFold(n_splits=3,random_state=0,shuffle=False)
    grid_search = GridSearchCV(RandomForestClassifier(), dictParaV, n_jobs=1, verbose=1,cv=cv,scoring='accuracy')
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    model = RandomForestClassifier(criterion=best_parameters['criterion'],
                max_depth=best_parameters['max_depth'], max_features=best_parameters['max_features'],
                min_samples_leaf=best_parameters['min_samples_leaf'],
                n_estimators=best_parameters['n_estimators'])
    model.fit(train_x, train_y)
    print (model)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y, paras):
    from sklearn import tree
    # paras是x,x,x,x模式字符串，最少含有一个null
    parArr = paras.split(',')
    if parArr[0] != 'null':
    # parArr参数待分配
        model = tree.DecisionTreeClassifier()
    else:
        model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y, paras):
    from sklearn.ensemble import GradientBoostingClassifier
    # paras是x,x,x,x模式字符串，最少含有一个null
    parArr = paras.split(',')
    if len(parArr)==7:
    # parArr参数待分配,由于是字符串数字，需要转化对应的值
    #     print 'NEW ParaM'
        max_leaf_nodesV=parArr[2]
        min_samples_leafV = parArr[4]
        random_stateV = parArr[5]
        max_featuresV = parArr[6]
        if(max_leaf_nodesV.lower()!='none'):#max_leaf_nodes为None或者整数
            max_leaf_nodesV=int(max_leaf_nodesV)
        else:max_leaf_nodesV=None
        if(float(min_samples_leafV)<1.0):#min_samples_leaf为小于1的float表示比例。或者整数表示个数
            min_samples_leafV=float(min_samples_leafV)
        else:
            min_samples_leafV = int(min_samples_leafV)
        if(random_stateV.lower()!='none'):#随机种子不为默认None则float化这个数
            random_stateV=int(random_stateV)
        else:random_stateV=None
        if(max_featuresV.lower()!='none'):
            if(float(max_featuresV)<1.0):
                max_featuresV=float(max_featuresV)
            else:
                max_featuresV = int(max_featuresV)
        else:max_featuresV=None
        model = GradientBoostingClassifier(n_estimators=int(parArr[0]),learning_rate=float(parArr[1]),max_leaf_nodes=max_leaf_nodesV,max_depth=int(parArr[3]),
                min_samples_leaf=min_samples_leafV,random_state=random_stateV,max_features=max_featuresV
                                           )
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1)
    print (train_x.shape,train_y.shape)
    model.fit(train_x, train_y)

    return model


# SVM Classifier
def svm_classifier(train_x, train_y, paras):
    from sklearn.svm import SVC
    # paras是x,x,x,x模式字符串，最少含有一个null
    parArr = paras.split(',')
    if parArr[0] != 'null':
    # parArr参数待分配
        model = SVC(kernel='rbf', probability=True)
    else:
        model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y, paras):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1,scoring='roc_auc')
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print (para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':

    startT = int(time.time() * 1000)
    datetime = 0
    username = ''
    meanAuc = 0.0
    aucf = ''
    prSavePath = ''
    rstSavePath = ''
    multibin = 0
    kappaScore = 0.0
    cmf = ''
    taskStatus = 1
    connection = None
    args = sys.argv
    isAutoAjust=False
    for item in args:
        print (':::', item)

    if len(args)  <6:
        print ('ERROR:paras is not enough! program aborted!')
        exit(1)
    elif args[3]=='FE':

        trainFPath = args[1]
        predictRstPath = args[2]
        paras = args[5]
    else:
        trainFPath = args[1]
        testFPath = args[2]
        predictRstPath = args[3]
        label = args[4]
        trainSize = args[5]
        paras = args[6]
        if len(paras.split(';'))>1:
            isAutoAjust=True


    username=os.path.split(os.path.split(predictRstPath)[0])[1]
    probarList=[startT, 5, taskStatus, username]
    probarTuple=tuple(probarList)
    sqlProgressBarInit = "INSERT INTO `ai`.`user_progressbar` (`dateid`, `barValue`, `taskStatus`, `userName`) VALUES (%d, %d, %d, '%s')" %probarTuple
    print (sqlProgressBarInit)
    mysqlProBar(sqlProgressBarInit)
    if args[3]=='FE':
        print (trainFPath, predictRstPath, paras)
        featureExtract(trainFPath, predictRstPath, paras)
    classifiers_label = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM','SVMCV','RFCV','GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'RFCV': random_forest_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    try:
        train_X, train_y, abc = getData_2(trainFPath)
        sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (30, startT)
        mysqlProBar(sqlProgressBarUpdate)
        test_X, test_y,abc = getData_2(testFPath)
        sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (50, startT)
        mysqlProBar(sqlProgressBarUpdate)
    except Exception as e:
        print (e.message)
        taskStatus = 0
    class_names = np.unique(train_y);
    is_binary_class = (len(class_names) == 2)
    if (str(testFPath).endswith('xlsx')):
        testX = pd.read_excel(testFPath, sep='\t')
    elif str(testFPath).endswith('csv'):
        testX = pd.read_csv(testFPath)
    elif str(testFPath).endswith('.sql'):
        flist = open(testFPath).readlines()
        count = len(flist)
        if count != 6:
            print ('sql file require 5 lines,only support 1 sql qurey.')
            exit(1)
        hostN = flist[0].split('=')[1].strip()
        portN = flist[1].split('=')[1]
        dbType = flist[2].split('=')[1].strip()
        userN = flist[3].split('=')[1].strip()
        passWd = flist[4].split('=')[1].strip()
        sqlStr = flist[5].strip()
        conn = None
        print (hostN, portN, dbType, userN, passWd, sqlStr)
        if (dbType == "hive"):
            conn = hive.Connection(host=str(hostN), port=int(portN), username=str(userN), password=str(passWd),
                                   auth='LDAP',
                                   thrift_transport=None)
        elif (dbType == "mysql"):
            conn = mysql.connector.connect(host=str(hostN), port=int(portN), user=str(userN), password=str(passWd),
                                           charset='utf8')
        elif (dbType == "oracle"):
            conn = cx_Oracle.connect(str(userN), str(passWd), str(hostN) + ":" + portN + "/orcl")
        df = pd.read_sql(sqlStr, conn)
        conn.close()
    model = None
    print (label+'<--label')
    classifier = None
    if label in classifiers_label:
        model = classifiers[label](train_X, train_y, paras )
    else:
        print ('Label %s is not support!!' % label)
        exit(1)
    if is_binary_class and not isAutoAjust:
        print ("***&&&****")
        multibin = 1
        meanAuc, model = getMeanAucBestModel(label, train_X, train_y)
        print ("meanAUC: ", meanAuc)
        print ("model: ", model)
    if is_binary_class:
        try:
            drawAUC(model, train_X, train_y)
            sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (65, startT)
            mysqlProBar(sqlProgressBarUpdate)
        except Exception as e:
            print (e.message)
        try:
            drawPR(model, train_X, train_y)
        except Exception as e:
            print (e.message)
    # else:
    #     meanAuc, model = getMeanAucBestModel(label, train_X, train_y)

    try:
        predict1 = model.predict(test_X)
        print (predict1.shape,'predict1.shape')
        writeResult(predict1, predictRstPath)
        sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (80, startT)
        mysqlProBar(sqlProgressBarUpdate)
        prec = precisionTest(trainFPath, label, paras )
        sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `barValue`=%d WHERE `dateid`=%d" % (100, startT)
        mysqlProBar(sqlProgressBarUpdate)
    except Exception as e:
        print (e.message)
        taskStatus = 0
    endT = int(time.time() * 1000)
    timeline=endT - startT
    print ('Total spent time: ' + str(timeline) + 'mm')
    sqlist = [startT, datetime, username, prec, meanAuc, aucf, prSavePath, trainFPath, testFPath, rstSavePath, label,trainSize,
              paras
              , multibin, kappaScore, cmf, taskStatus,timeline]

    sqlTuple = tuple(sqlist)
    print (sqlTuple)
    print (aucf)
    if not taskStatus:
        sqlProgressBarUpdate = "UPDATE `ai`.`user_progressbar` SET `taskStatus`=%d WHERE `dateid`=%d" % (0, startT)
        mysqlProBar(sqlProgressBarUpdate)
    sqlstr = "insert into webai_info (dateid,datetime,username,acc,auc,aucf,pr,trainf,predictf,resultf,label,trainSize,paras,multibin,kappaScore,cmf,taskStatus,timeline_millisecond) values (%d,%d,'%s',%.4f,%.4f,'%s','%s','%s','%s','%s','%s','%.4f','%s',%d,%.4f,'%s',%d,%d)" % sqlTuple
    rst = ToMysql(sqlstr)
    print('RESULT')

