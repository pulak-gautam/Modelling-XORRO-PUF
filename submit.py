import numpy as np
import time as tm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
 
#    ignoring convergence warnings of sklearn
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

#    number of XORROs
N = 16 

##   helper functions:
def createFeatures(dataset):
#     creates a 2d numpy array with #columns = 64(ai) + 64(1-ai) + 1 (yi) = 129 and #rows = #rows(dataset) 
    X = dataset[:,:64]
    Y = dataset[:,-1:]
    return np.concatenate((X, 1 - X, Y), axis=1) 

def createAandB(dataset):
#     creates upper xorro and lower xorro pair for the given dataset 
#     from columns (64-67) and (68-71) respectively
    X = 8*dataset[:, 64] + 4*dataset[:, 65] + 2*dataset[:, 66] + dataset[:, 67]
    Y = 8*dataset[:, 68] + 4*dataset[:, 69] + 2*dataset[:, 70] + dataset[:, 71]

    X = np.reshape(X, (len(X), 1))
    Y = np.reshape(Y, (len(Y), 1))
    
    return np.concatenate((X, Y), axis=1).astype(int)
def createX(i, j):
    temp = [0 if (k<i or k>=j) else 1 for k in range(0, N-1)]
    return np.array(temp)

def createX2(i, j):
    temp = np.zeros((1,15))
    temp[0, i-2]=-1
    temp[0, j-2]=1
    return temp

def cleanData(dataset, AB):
    data = [[np.zeros((1,129)) for i in range(N)] for j in range(N)]
    
#     handling A > B
#     1. handles yi change
    isLarger = AB[:,0] > AB[:, 1]
    isLarger = np.reshape(isLarger, (len(isLarger), 1))
    isLarger = isLarger.astype(int)
    dataset[:,-1] = np.bitwise_xor(isLarger[:, -1],(dataset[:, -1].astype(int)))
    
#     2. handles swapping A and B
    isLarger = AB[:,0] > AB[:, 1]
    copyMatrix = np.copy(AB[:, 0])
    AB[isLarger, 0] = AB[isLarger, 1]
    AB[isLarger, 1] = copyMatrix[isLarger]

#     divides the dataset based on AB matrix into smaller datasets of 
#     inputs of appropriate dimension (#columns = 129) being fed to the same xorro pair 
    for index in range (0, len(dataset)):
        if(AB[index, 0] == AB[index, 1]):
            continue
            
        data[AB[index, 0]][AB[index, 1]] = np.concatenate((data[AB[index, 0]][AB[index, 1]],
                        np.reshape(dataset[index,:], (1, len(dataset[index, :])))), axis=0)

    for i in range (0, N):
        for j in range (i+1, N):
            data[i][j] = np.delete(data[i][j], (0), axis=0)
            
    return data

def transformData(Z_train):
#     transforms the dataset to feed it into my_fit()
    train_features = createFeatures(Z_train)
    AB_train = createAandB(Z_train)
    return cleanData(train_features, AB_train)

def training(dataset):
#     trains the dataset and returns a model for the given data
#     requires the dataset to be strictly of the form of a numpy array

#     change hyperparameters: loss, penalty, etc.
#     choose model

    clf = LinearSVC(loss = "squared_hinge", C=0.2, tol=1)
    # clf = LogisticRegression(C=5,solver="sag")

    clf.fit(dataset[:,:-1], dataset[:,-1])
    return clf

def refineModel(model):
#   uses the relation, let's say R, that alpha(i,j) can be written as the sum of alpha(k, k+1), 
#   where k ranges from i to j-1. the relation R holds true for beta
#   since model(i, j) is composed of alphas and betas, the relation R holds for model(i, j) as well

    X = np.array(np.reshape(createX(0, 2), (1, len(createX(0, 2))))) 
    temp = np.array(model[0][2].coef_)
    Y = np.reshape(temp, (1, len(temp[0])))
    
    for i in range (0, N):
        for j in range (i+2, N):
            if(i==0 and j==2):
                continue
            temp = np.array(model[i][j].coef_)
            temp = np.reshape(temp, (1,len(temp[0])))
            X = np.concatenate((X, np.array(np.reshape(createX(i, j), (1, len(createX(i, j)))))), axis=0)
            Y = np.concatenate((Y, temp), axis=0)
            
    W = LinearRegression().fit(X, Y).coef_
    index = 0

    for i in range (0, N):
        for j in range (i+2, N):
            tmp = np.dot(W, X[index])
            model[i][j].coef_ = np.reshape(tmp, np.shape(model[i][j].coef_))
            index+=1

    return model

def refineModel2(model):
#   instead of using the relation mentioned above, we define a relation G using relation R:
#   model(i, j) = model(0,j) - model(0, i)

    X = np.array(np.reshape(createX(1, 2), (1, len(createX(1, 2))))) 
    temp = np.array(model[1][2].coef_)
    Y = np.reshape(temp, (1, len(temp[0])))
    
    for i in range (1, N):
        for j in range (i+1, N):
            if(i==1 and j==2):
                continue
            temp = np.array(model[i][j].coef_)
            temp = np.reshape(temp, (1,len(temp[0])))
            X = np.concatenate((X, np.array(np.reshape(createX(i, j), (1, len(createX(i, j)))))), axis=0)
            Y = np.concatenate((Y, temp), axis=0)
            
    W = LinearRegression().fit(X, Y).coef_
    index = 0

    for i in range (1, N):
        for j in range (i+1, N):
            tmp = np.dot(W, X[index])
            model[i][j].coef_ = np.reshape(tmp, np.shape(model[i][j].coef_))
            index+=1

    return model

################################
# Non Editable Region Starting #
################################
def my_fit(Z_train):
################################
#  Non Editable Region Ending  #
################################

#     creates a model array, each element of which represents a model for (i, j)th xorro pair, given i<j
    model = [[None for i in range(N)] for j in range(N)]
    train_ = transformData(Z_train)
    
    for i in range (0, N):
        for j in range (i+1, N):
            model[i][j] = training(train_[i][j])

#     refines the model using linear regression    
    refineModel2(model)
    refineModel(model)
    return model

 ################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, model):
################################
#  Non Editable Region Ending  #
################################

#     predicts the output (yi) for a given test input (#columns = 73) based on the model given as the input parameter
#     returns a numpy array y_predict (#rows = #rows(Z_tst) and #columns = 1)

    AB_train = createAandB(X_tst)
    new_tst = np.concatenate((X_tst[:,:64], 1-X_tst[:,:64]), axis = 1)
    y_predict = np.zeros((len(X_tst),1))
    
    for index in range(0, len(X_tst)):
        a = AB_train[index, 0]
        b = AB_train[index, 1]
        
        if(a>b):
            y_predict[index,0] = 1 - model[b][a].predict(np.reshape(new_tst[index,:],(1, len(new_tst[index]))))
        else:
            y_predict[index,0] = model[a][b].predict(np.reshape(new_tst[index,:],(1, len(new_tst[index]))))
            
    return np.reshape(y_predict,(1,(len(y_predict))))
            
    
#     main
if __name__ == '__main__':
    Z_trn = np.loadtxt( "train.dat" )
    Z_tst = np.loadtxt( "test.dat" )

    start_time = tm.time()
    pred = my_predict(Z_tst, my_fit(Z_trn))
    end_time = tm.time()

    acc = np.average(Z_tst[ :, -1 ] == pred)

    print(acc)
    print("time taken:", (end_time - start_time), "sec")