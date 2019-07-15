from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
import numpy as np
from model import SvmClassifierModel
from sklearn.model_selection import train_test_split
from os import listdir
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

# define image extractor function
def image__features_extractor(directory):
    for img in listdir(directory):
        print(img)
        filename = directory + '/' + img
        image = load_img(filename, target_size=(28, 28))
        image = img_to_array(image)
    return image

def classifyOne():
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

    # call model
    SVmClassifier = SvmClassifierModel()
    # fit data into LRclassifier
    SVmClassifier.fit(Xtrain, Ytrain)
    # evaluate
    predicted = SVmClassifier.predict(Xtest)
    Ytest11 = list()
    for elt in Ytest:
        elt = int(elt)
        Ytest11.append(elt)
    print('accuracy : {}%'.format(str(accuracy_score(Ytest11, predicted)*100)))
    print(classification_report(Ytest11, predicted))
    print(confusion_matrix(Ytest11, predicted))
    # rescale and reshape image
    input_image = 'image_to_recognize'
    image = image__features_extractor(input_image)

    image = np.resize(image, (1, 784))
    image = np.array(image).reshape((1, -1))
    lda = LDA(n_components=10)
    image = lda.transform(image)
    print('image shape : ', image.shape)
    # prediction
    predicted = SVmClassifier.predict(image)

    image2 = image__features_extractor(input_image)
    image2 = np.resize(image2, (1, 784))
    image2 = np.array(image2).reshape((1, -1))
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.title("Predicted : {}".format(predicted))
    print('predicted : ', predicted)
    plt.imshow(np.reshape(image2, (28, 28)), cmap='gray_r')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    classifyOne()

"""
def classify():
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
    #fit data into LRclassifier
    #XtrainA = np.concatenate(Xtrain, Xtrain1) YtrainA = np.concatenate(Ytrain, Ytrain1)
    SVmClassifier.fit(Xtrain, Ytrain)

    #predict result
    predicted = SVmClassifier.predict(Xtest)
    print('predicted : ', [int(predict) for predict in predicted])
    print('Ytest1 : ', Ytest)
    print('mean error : ', np.mean(predicted - Ytest)**2)
    #accuracy
    print(classification_report(Ytest, [int(predict) for predict in predicted]))
    print(confusion_matrix(Ytest, [int(predict) for predict in predicted]))

    plt.figure(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.title("Predicted : {}".format(predicted[i]))
        plt.imshow(np.reshape(Xtest[i], (28, 28)), cmap='gray_r')
        plt.tight_layout()
    plt.show()
"""