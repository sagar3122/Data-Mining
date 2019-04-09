import numpy
import copy
import matplotlib.pyplot as plt
import itertools

mean1 = [1,0]
mean2 = [0,1.5]
cov1 = [[0.9,0.4],[0.4,0.9]]
cov2 = [[0.9,0.4],[0.4,0.9]]
sample1 = numpy.random.multivariate_normal(mean1, cov1, 500)
sample2 = numpy.random.multivariate_normal(mean2, cov2, 500)
# print(sample1.shape)
# print(sample1)
# print("\nBreak\n")
# print(sample2)
# print(sample2.shape)
data_set = numpy.concatenate((sample1,sample2))
# print(data_set)
# print(len(data_set))
# print(data_set.shape)
val_K = input("Enter the no of clusters: ")
k = int(val_K)
c = []
# centers = []
for x in range(k):
    val_centers = input("Enter the centers for cluster "+ str(x+1) +" : ")
    # print(val_centers)
    # print(type(val_centers))
    centers = []
    for x in val_centers.split(" "):
        # print(x)
        centers.append(float(x))
    #print(centers)
    c.append(centers)
c = numpy.asarray(c, dtype = numpy.float64)
#print(c)
#print(c.shape)
def myKmeans(X,k,c):
    #num of attributes in the data set
    num_attributes = data_set.shape[1]
    # num of data points in the data set
    len_data = data_set.shape[0]
    #intialize the ndarray containing 0's with the shape same as c for tracking the old centers
    old_c = numpy.zeros(c.shape)
    #current centers we work on
    new_c = copy.deepcopy(c)
    #ndarray representing the cluster num to which each data point belongs to, intialized with 0's
    clusters = numpy.zeros(len_data)
    #ndarray with subarrays which represent the data points in the data set
    #and the k elements in the sub arrays represent the distance of that data point from the k clusters
    distance = numpy.zeros((len_data,k))
    #stopping condition, computing the L2 norm between the new centers and old centers
    stop = numpy.linalg.norm(new_c-old_c)
    #loop until the l2 norm after the updation of the centers is <=0.001 or iterations reach 10000
    i = 0
    while stop > 0.001:
        i = i+1
        print("Iteration: ",i)
        #check for the number of iterations
        if i == 10000:
            break
        #compute the distance between every data point and center for all clusters using l2 norm
        for x in range(k):
            distance[:, x] = numpy.linalg.norm(data_set - new_c[x], axis = 1)
        #assign the data points to the closest center of a cluster
        clusters = numpy.argmin(distance, axis = 1)
        print(clusters)
        #print(clusters.shape)
        #update the old center points
        old_c = copy.deepcopy(new_c)
        #compute the new center points by calculating the mean values of the data points for a cluster
        for x in range(k):
            new_c[x] = numpy.mean(data_set[clusters == x], axis = 0 )
        #recompute the stopping criterion
        stop = numpy.linalg.norm(new_c - old_c)
        print(stop)
    print("Centers after kmeans converges: ", new_c)

    #scatter plot
    xplots = []
    yplots = []
    for x in data_set:
        for d in x:
            if d == x[0]:
                xplots.append(d)
            if d == x[1]:
                yplots.append(d)
    #print("xplots :",xplots)
    #print("yplots :",yplots)
    # color = [red, blue, green, yellow]
    for p,x,y in zip(clusters, xplots, yplots):
        if p == 0:
            plt.scatter(x, y, color = 'c')
            continue
        elif p == 1:
            plt.scatter(x, y, color = 'g')
            continue
        elif p == 2:
            plt.scatter(x, y, color = 'b')
            continue
        elif p == 3:
            plt.scatter(x, y, color = 'y')
            continue

    for x,y in new_c:
        plt.scatter(x, y, color = 'r')

    # plt.scatter(xplots, yplots)
    plt.xlabel("xplots")
    plt.ylabel("yplots")
    plt.show()


myKmeans(data_set,k,c)
