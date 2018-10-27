from sklearn.linear_model import LinearRegression

def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    
    reg = LinearRegression()
    reg = reg.fit(ages_train, net_worths_train)
    
    
    return reg
