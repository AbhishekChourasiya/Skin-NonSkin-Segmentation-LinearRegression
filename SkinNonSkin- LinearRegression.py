
import csv
import numpy as np
from scipy import linalg as spl 
from PIL import Image
import matplotlib.pyplot as plt

def logRegress(X, Tx,y):
    
    #pseudoinverse
    Xp = np.linalg.pinv(X)    
 
    
    print("pseudo")
    print(Xp.shape)
    print(Xp)
 
    
    A = np.dot(Xp, y)
    
      
    
    
    Y = Tx
    Yt = Y.transpose()
    r = np.dot(A, Yt)
    #final values
    print("r",r)
    
    p = np.exp(r)/(1+np.exp(r))

    
    print("p shape",p.shape)


    print("p",p)
    return p
    
    
def main():
 
    dataset = np.genfromtxt('skin.csv',delimiter=',')
    classdataset =np.genfromtxt('skinclass.csv',delimiter=',')    
    #Creating Imageset as input.
    myImage= Image.open('newface.jpg').convert('RGBA').convert('RGB')
    myImage.show();
    imageset = np.array(myImage.getdata())
       
    #Generating Output Matrix.
    p =logRegress(dataset,imageset,classdataset)
    
    print("final PREDICTION")
    print(p)
    
    
    #Generating Output Image.
    outImage = Image.new(myImage.mode, myImage.size)
    outImageSet = list(outImage.getdata())
    for i in range(len(outImageSet)):
        if (p[i] <= 0.85):
            #print('yes')
            outImageSet[i] = (0,0,0,0)
        else:
            #print('no')
            outImageSet[i] = (255,255,255,0)
            
    outImage.putdata(outImageSet);
    outImage.show()
    outImage.save('out.jpg')
    outImage.close()

    
main()





