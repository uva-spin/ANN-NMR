import numpy as np

def Predict(X,testmodel):
    acc = []
    X.to_numpy()
    X = np.reshape(X,(1,500))
    Y = testmodel.predict(X)
    Y = Y.reshape((len(Y),))
    g = 0.05
    s = 0.04
    bigy=(3-s)**0.5

    def cosal(x,eps):
        return (1-eps*x-s)/bigxsquare(x,eps)


    def c(x):
        return ((g**2+(1-x-s)**2)**0.5)**0.5


    def bigxsquare(x,eps):
        return (g**2+(1-eps*x-s)**2)**0.5


    def mult_term(x,eps):
        return float(1)/(2*np.pi*np.sqrt(bigxsquare(x,eps)))


    def cosaltwo(x,eps):
        return ((1+cosal(x,eps))/2)**0.5


    def sinaltwo(x,eps):
        return ((1-cosal(x,eps))/2)**0.5


    def termone(x,eps):
        return np.pi/2+np.arctan((bigy**2-bigxsquare(x,eps))/((2*bigy*(bigxsquare(x,eps))**0.5)*sinaltwo(x,eps)))


    def termtwo(x,eps):
        return np.log((bigy**2+bigxsquare(x,eps)+2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps))/(bigy**2+bigxsquare(x,eps)-2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps)))

    def icurve(x,eps):
        return mult_term(x,eps)*(2*cosaltwo(x,eps)*termone(x,eps)+sinaltwo(x,eps)*termtwo(x,eps))
        
    xvals = np.linspace(-6,6,500)
    
    p_pred = Y
    r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
    center = 250
    length = range(500)
    norm_array = []
    for x in length:
        norm_array = np.append(norm_array,(x - center)*(12/500))  
    Iplus = icurve(norm_array,1)
    Iminus = icurve(norm_array,-1)
    array_pred = r_pred*Iminus
    array_pred_flipped = np.flip(array_pred)
    element_1_pred = array_pred_flipped + Iminus
    sum_array_pred = np.sum(array_pred_flipped)*(12/500)
    element_2_pred = 1/sum_array_pred
    element_3_pred = p_pred
    result_pred = element_1_pred*element_2_pred*element_3_pred
    result_pred_new = result_pred.reshape(500,)
    return p_pred, result_pred_new,xvals
