import matplotlib.pyplot as plt
class Simple_Linear_Regression:
    def __init__(self , learning_rate=0.001 ,visualize=True,m = 0, b = 0):
        self.learning_rate = learning_rate
        self.m = m 
        self.b = b
        self.visualize = visualize
        self.m_gradient_total = []
        self.b_gradient_total = []

    def fit(self,x,y,epochs):
        if len(x)!=len(y):
            print("No of values in X are not equal to No of values in Y")
            return
        for i in range(epochs):
            self._train_gradient_descent(x,y)
            if i%100 == 0 :
                print("Epoch == ",i)
                print("M - > ",self.m)
                print("B - > ",self.b)
                print("---------------------")
        if self.visualize:
            figure, axis = plt.subplots(2, 2)
            axis[0,0].scatter(x,y)
            axis[0,0].set_title("Data")

            axis[0,1].scatter(x,y)
            axis[0,1].plot(list(range(min(x)- 1,max(x)+2)),[self.m*x+self.b for x in range(min(x)- 1,max(x)+2)])
            axis[0,1].set_title("Regression Line ")

            axis[1,0].plot(list(range(0,epochs)),self.m_gradient_total)
            axis[1,0].set_title("M Loss")
            
            axis[1,1].plot(list(range(0,epochs)),self.b_gradient_total)
            axis[1,1].set_title("b Loss")
    
            plt.show()

    def predict(self,x):
        return (self.m*x + self.b)
    
    def _train_gradient_descent(self,x,y):
        m_gradient = 0 
        b_gradient = 0 
        n = len(x)
        for i in range(n):
            m_gradient += x[i]*(y[i]-(self.m*x[i]+self.b))
            b_gradient += (y[i]-(self.m*x[i]+self.b))
        m_gradient = -(2/n) * m_gradient
        b_gradient = -(2/n) * b_gradient
        self.m_gradient_total.append(m_gradient)
        self.b_gradient_total.append(b_gradient)
        self.m = self.m - (self.learning_rate*m_gradient)
        self.b = self.b - (self.learning_rate*b_gradient)