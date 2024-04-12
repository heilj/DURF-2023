from utils import *
import numpy as np
import matplotlib.pyplot as plt

CLASS1 = 'viral'
CLASS2 = 'quality'
CLASS3 = 'memoryless'
CLASS4 = 'junk'

do_classify = True

source = 'top'
current_dir = get_parrent()
file_path = f'{current_dir}/{source}_dao.csv'
data_file = read_file(file_path,False,False,thres=1000)

def plot(CLASS):
    if CLASS:
        data = classfy(CLASS, data_file, 0.3, source)
    else:
        data = data_file
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])
 
    scalar = RobustScaler()
    X = logt(independent_variables)
    X = scalar.fit_transform(X)
    Y = logt(dependent_values)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter([i[6] for i in X], Y, c='r', s=10.0, alpha=0.3)
    plt.xlabel('logt x')
    plt.ylabel('logt y')
    ax.grid()

    # Set x/y axis limits
    plt.show()

    plt.hist(Y, bins=500)
    plt.ylabel('frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot([1,2,3,4,5,6,7],independent_variables[0])

    plt.xlabel('Days')
    plt.ylabel('views after logt and scalar')
    
    plt.figure(figsize=(10, 6))
    plt.plot([1,2,3,4,5,6,7],X[0])

    plt.xlabel('Days')
    plt.ylabel('views after logt and scalar')
    

    plt.show()


plot(CLASS1)