from sklearn.model_selection import train_test_split
from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from model import SvmClassifierModel
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

def train():
    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    df_test_x, df_test_y, df_train_x, df_train_y = load_data_from_csv()
    Xtrain2, Xtest2a, Ytrain2, Ytest2a = train_test_split(df_train_x, df_train_y, random_state=0, test_size=0.9)
    Xtrain2b, Xtest2, Ytrain2b, Ytest2 = train_test_split(df_test_x, df_test_y, random_state=0, test_size=0.9)
    df_train_xM, df_train_yM = load_data_from_mongoDb()
    Xtrain3, Xtest3, Ytrain3, Ytest3 = train_test_split(df_train_xM, df_train_yM, random_state=0, test_size=0.9)

    #agregate data with numpy
    Xtrain, Ytrain, Xtest, Ytest = agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3)
    #reduce dimension of agregated data with PCA
    Xtrain, Xtest = reduction_of_dimension_with_PCA(Xtrain, Xtest)
    # reduce dimension of agregated data with LDA
    #Xtrain, Xtest = reduction_of_dimension_with_LDA(Xtrain, Xtest, Ytrain)

    #call model
    SVmClassifier = SvmClassifierModel()
    #fit data into RFClassifier
    SVmClassifierTrain=SVmClassifier.fit(Xtrain, Ytrain)
    print('train score : {} %'.format(SVmClassifier.score(Xtest, Ytest) * 100))
    #print('intercept : {}'.format(LRClassifier.intercept_))
    #print('coeff : {}'.format(LRClassifier.coef_))

if __name__=='__main__':
    train()