import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator

class BaseFSModel(BaseEstimator):
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self  

    def s_fi(self, fi, x):
        re1 = (x - fi)
        re1 = re1* (re1>0).astype(float)
        re2 = (-1*x - fi)
        re2 = re2* (re2>0).astype(float)
        re = re1 - re2
        return re

    def reshapeInput(self, X):
        XList = list()

        n_list  =[0,sum(num_features[:1]),sum(num_features[:2]),sum(num_features)]
        for i in range(len(n_list) -1):
            XList.append(X[:, n_list[i]:n_list[i+1] ])
        return XList

class ADLSL(BaseFSModel):
    def __init__(self, alpha = 1, beta=1,  gamma = 1,PI_1 = 1,PI_2=1,PI_3=1,EPS=3,  h = 1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.h = h
        self.PI_1 = PI_1
        self.PI_2 = PI_2
        self.PI_3 = PI_3
        self.EPS = EPS

    def get_params(self, deep=True):
        return {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'h': self.h,
                'PI_1': self.PI_1,
                'PI_2': self.PI_2,
                'PI_3': self.PI_3,
                'EPS': self.EPS
                }

    def fit(self, X,Y):
        X = self.reshapeInput(X)
        self.train_X = X

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        h = self.h
        PI_1 = self.PI_1
        PI_2 = self.PI_2
        PI_3 = self.PI_3
        PI = self.EPS

        piList = (PI_1,PI_2,PI_3)

        n_ite = 200
        n_views = len(X)


        lamda = np.ones((n_views)) / n_views
        n_samples = X[0].shape[0]

        UList = list()
        PList = list()
        SList = list()
        LsList = list()
        bList = list()
        HList = list()
        AList = list()
        CList = list()
        H = np.random.randn(n_samples, int(h))
        for i_view in range(n_views):
            UList.append(X[i_view] * 1)
            PList.append(np.random.randn(X[i_view].shape[1], h))
            bList.append(np.zeros((h,1)))
            S = np.ones((X[i_view].shape[1],X[i_view].shape[1]))
            S = S /X[i_view].shape[1]
            S = S*S
            SList.append( (S +S.transpose())/2 )
            D_ = np.diag(np.sum(S, axis=1))
            LsList.append( D_ - SList[i_view])
            HList.append( H*1)
            AList.append( np.zeros_like(X[i_view]) )
            CList.append( np.zeros_like(X[i_view]) )

        rho = 1e-4

        lossP = 0
        for ite in range(n_ite):
            # print(ite)
            loss1 = 0
            loss2 = 0
            loss3 = 0
            loss4 = 0
            for i_view in range(n_views):
                if i_view!=n_views -1 :
                    loss1 += (lamda[i_view]**PI) * np.sum( np.linalg.norm(X[i_view] - UList[i_view] )**2 )
                else:
                    loss1 += (lamda[i_view]**PI) * np.sum(np.abs(X[i_view] - UList[i_view]) )
                loss2 += (lamda[i_view]**PI) * np.trace( np.matmul(np.matmul(UList[i_view], LsList[i_view]),UList[i_view].transpose()) )
                loss3 += (lamda[i_view]**PI) * np.linalg.norm( H -np.matmul(np.ones((n_samples, 1)),bList[i_view].transpose()) -  np.matmul(UList[i_view], PList[i_view]) )**2
                PNorm = np.linalg.norm(PList[i_view], axis = 1)
                loss4 += (lamda[i_view]**PI) * np.sum( ( (1 + piList[i_view]) * PNorm * PNorm  ) /(PNorm + piList[i_view] ))
            lossC = loss1 + gamma * loss2 + alpha * loss3 + beta *loss4


            if  ite > 10:
                flaG = 0
                for i_view in range(n_views):
                    if np.linalg.norm(X[i_view] - UList[i_view] - AList[i_view], ord = np.inf) >1e-4:
                        flaG = 1
                if flaG == 0:
                    break


            # Update U
            for i_view in range(n_views):
                if i_view !=n_views -1 :
                    U1 = X[i_view] + alpha *np.matmul(HList[i_view] ,PList[i_view].transpose())
                    U2 = np.eye(X[i_view].shape[1]) + 2 * gamma * LsList[i_view] + alpha * np.matmul(PList[i_view], PList[i_view].transpose())

                    c, low = cho_factor(U2.transpose())
                    U2   = cho_solve((c, low), U1.transpose())
                    UList[i_view] = U2.transpose()

                else:
                    U1 = 2*alpha *np.matmul(HList[i_view],PList[i_view].transpose()) +  (X[i_view]-AList[i_view]+CList[i_view] / rho) * rho
                    U2 = 4*gamma*LsList[i_view]+2*alpha*np.matmul(PList[i_view],PList[i_view].transpose()) + np.eye(PList[i_view].shape[0]) * rho


                    c, low = cho_factor(U2.transpose())
                    U2   = cho_solve((c, low), U1.transpose())
                    UList[i_view] = U2.transpose()

                    AList[i_view] = self.s_fi(1/  rho, X[i_view] - UList[i_view] + CList[i_view]/ rho)
                    CList[i_view] = CList[i_view] +(X[i_view] - UList[i_view]- AList[i_view]) * rho

            rho = min(1e6,1.1*rho)

            # # Update S
            for i_view in range(n_views):
                Sd = cdist(UList[i_view].transpose(), UList[i_view].transpose())
                Sd = 1/(Sd*Sd +1e-12)
                Sd = Sd * (np.ones( (Sd.shape[0], Sd.shape[0])) - np.eye(Sd.shape[0]))
                S = Sd / np.tile(np.sum(Sd,axis = 1).reshape(-1,1),(1,Sd.shape[0]))
                S = S * S

                S = (S +S.transpose())/2
                SList[i_view] =  S*1
                D_ = np.diag(np.sum(S, axis=1))
                LsList[i_view] = D_ - S



            # Update P
            for i_view in range(n_views):
                wi= np.linalg.norm(PList[i_view], axis = 1)
                P23 = np.diag(0.5*(1+piList[i_view])*(wi + 2 *piList[i_view])/((wi+piList[i_view])**2 +1e-12 ))
                P1 = np.matmul(UList[i_view].transpose(), UList[i_view]) + beta / alpha * P23
                P2 = np.matmul( UList[i_view].transpose(), HList[i_view])
                c, low = cho_factor(P1)
                PList[i_view]   = cho_solve((c, low), P2)


            # Update b
            for i_view in range(n_views):
                bList[i_view] = np.matmul(H.transpose() - np.matmul(PList[i_view].transpose(), UList[i_view].transpose()), np.ones((n_samples, 1))) / n_samples

            # Update H
            for i_view in range(n_views):
                if i_view==0:
                    U1 = (lamda[i_view]**PI) * (np.matmul(UList[i_view],PList[i_view]) +np.matmul(np.ones( (n_samples, 1)),bList[i_view].transpose()))
                else:
                    U1 +=  (lamda[i_view]**PI) * (np.matmul(UList[i_view],PList[i_view])+np.matmul(np.ones((n_samples, 1)),bList[i_view].transpose()))
            U2,_,U3 = np.linalg.svd(U1)
            U2 = U2[:, 0:h]
            H = np.matmul(U2, U3)

            for i_view in range(n_views):
                HList[i_view] = H - np.matmul(np.ones((n_samples, 1)),bList[i_view].transpose())


            # Update lamda
            dlist = list()
            for i_view in range(n_views):
                if i_view!=n_views-1 :
                    loss1 = np.sum( np.linalg.norm(X[i_view] - UList[i_view] )**2 )
                else:
                    loss1 = np.sum(np.abs(X[i_view] - UList[i_view]) )
                loss1 += gamma * np.trace( np.matmul(np.matmul(UList[i_view], LsList[i_view]),UList[i_view].transpose()) )
                loss1 += alpha * np.linalg.norm( H -np.matmul(np.ones((n_samples, 1)),bList[i_view].transpose()) - np.matmul(UList[i_view], PList[i_view]) )**2
 
                PNorm = np.linalg.norm(PList[i_view], axis = 1)
                loss1 += beta * (lamda[i_view]**PI) * np.sum( ( (1 + piList[i_view]) * PNorm * PNorm  ) /(PNorm + piList[i_view] ))

                dlist.append( ( PI*loss1) ** (1/(1 - PI)) )
            for i_view in range(n_views):
                lamda[i_view] = (dlist[i_view]/sum(dlist))

        row_norms = np.linalg.norm(PList[0], axis=1)
        top_20_indices = np.argsort(row_norms)[-10:]
        for idx in top_20_indices:
            feature_count0[idx] += 1

        row_norms = np.linalg.norm(PList[1], axis=1)
        top_20_indices = np.argsort(row_norms)[-10:]
        for idx in top_20_indices:
            feature_count1[idx] += 1

        row_norms = np.linalg.norm(PList[2], axis=1)
        top_20_indices = np.argsort(row_norms)[-10:]
        for idx in top_20_indices:
            feature_count2[idx] += 1

        self.P = PList

num_features = [310, 106, 1961]
feature_count0 = np.zeros(num_features[0])
feature_count1 = np.zeros(num_features[1])
feature_count2 = np.zeros(num_features[2])


Noise = 0
if __name__ == "__main__":
    # Loading data
    dataName = 'ImgGendata_GO23'
    X, W  = load_sysdata(dataName)
    n_views = len(X)

    X = np.concatenate(X, axis = 1)

    params = {
        'alpha':  100.0,
        'beta':  1000000.0,
        'gamma':  10.0,
        'h':     5,
        'PI_1': 1,
        'PI_2': 1,
        'PI_3': 1,
        'EPS':  5,
        }

    modelT = ADLSL()
    modelT.set_params(**params)
    modelT.fit(X, W)

    print(np.argsort(feature_count0)[-10:])
    print(np.argsort(feature_count1)[-10:])
    print(np.argsort(feature_count2)[-10:])

    print(np.sort(feature_count0)[-10:])
    print(np.sort(feature_count1)[-10:])
    print(np.sort(feature_count2)[-10:])
