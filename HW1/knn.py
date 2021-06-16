import numpy as np
import statistics
import sys
import csv
import statistics

test_file = 'test_pub.csv'
train_file = 'train.csv'
filename = "result_data.csv"

#Skip over the first row
def gatherData(file_name):
    data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
    return data

#Euclidean distance, Q6 utilizes linalg.norm
def retrieve_neighbors(tr_data, test_point, neighbors):
    return np.argsort(np.linalg.norm(tr_data - test_point, axis=1))[:neighbors]

#Q7 KNN sorting happens
def knn_sort(tr_data, test_point, neighbors):
    return int(statistics.mode([tr_data[i, 86] for i in retrieve_neighbors(tr_data[:, 1:85], test_point[1:85], neighbors)]))

def retrieve_performance(te_data, tr_data, neighbors):
    sameC = 0
    total_points_looked = 0
    for i in te_data:
        if knn_sort(tr_data, i, neighbors) == int(i[86]):
            sameC = sameC + 1
        total_points_looked = total_points_looked + 1
    performance_rating = sameC / total_points_looked
    return performance_rating

def classify(tr_data, te_data, neighbors):
    print("id, income")
    printlist = []
    for i in te_data:
        print(int(i[0]), int(knn_sort(tr_data, i, neighbors)))
        #printlist = [(int(i[0]), int(knn_sort(tr_data, i, neighbors)))]

#Divide the data up into four different fold segments
#Since we're looking at 8000 entries of data it would make the most sense
#to split it up by 2000s into four different segments
#Question #8 implementination of 4-fold cross validation
def fourFoldcross(data, neighbors):
    crossValidation_results = list()
    #First this function evaluates the performance using the first 6000 rows as its own training set 
    # Then it uses rows 6000 - 8000 as its testing set
    crossValidation_results.append(retrieve_performance(data[:int(6000)], data[(int(6000)):], neighbors))

    #Second, this function evaluates the rows between 0 and 4000, then 6,000 - 8,000 as its own training set
    #And testing the rows, 4000 - 6000
    crossValidation_results.append(retrieve_performance(np.concatenate((data[:int(4000)], data[int(6000):8000]), axis = 0), data[int(4000):int(6000)], neighbors))

    #Thirdly this function evaluates the rows between 0 to 2000 then 4,000 to 8,000 as its own training set , 
    # Then it uses the 4000 - 6000 rows as its testing set
    crossValidation_results.append(retrieve_performance(np.concatenate((data[:int(2000)], data[int(4000):8000]), axis = 0), data[int(4000):int(6000)], neighbors))

    #Lastly it evaluates the rows between 2000 to 8000 inclusively as its training set and then
    # Tests the rows between 0 to 2000 incuslively as its testing set
    crossValidation_results.append(retrieve_performance(data[(int(2000)):], data[:int(2000)], neighbors))
    return crossValidation_results
    
if __name__ == "__main__":
    tr_data = gatherData(train_file)
    te_data = gatherData(test_file)
    Q9_results_file = open("hyperP.txt", "a")
    #classify(tr_data, te_data, 12)
    hyperParameterList = [1,3,5,7,9,99,999,8000]
    for i in hyperParameterList:
    #    print("\n")
    #    print("Neighbors have been set to :  " + str (i))
    #    print("Performance measurements : \n")
    #    print("____________________________________________________\n")
    #    print("Fold 1 observed accuracy rating : \n" + (str)(fourFoldcross(tr_data, i)[0]) )
    #    print("Fold 2 observed accuracy rating : \n" + (str)(fourFoldcross(tr_data, i)[1]) )
    #    print("Fold 3 observed accuracy rating : \n" + (str)(fourFoldcross(tr_data, i)[2]) )
    #    print("Fold 4 observed accuracy rating : \n" + (str)(fourFoldcross(tr_data, i)[3]) )
    #    print("____________________________________________________\n")

    #This is for question 9, it builds a file that tests through each of the
    #elements within the hyperParameter list above.
    # This takes a very long time, but it outputs the correct data in a very neat fashion.
        #mean = fourFoldcross(tr_data, i) / 4
        #print("Mean calculation : " + mean)
        Q9_results_file.write("Testing performance on differing Ks : \n")
        Q9_results_file.write("Neighbors have been set to :  " + str (i))
        Q9_results_file.write("\n")
        Q9_results_file.write("Performance percentage (Training Accuracy): ")
        Q9_results_file.write(str (retrieve_performance(tr_data, tr_data, i)))
        Q9_results_file.write("\n")
        Q9_results_file.write("Validation Accuracy rating : ")
        mean = ((fourFoldcross(tr_data, i)[0]) + (fourFoldcross(tr_data, i)[1]) + (fourFoldcross(tr_data, i)[2]) + (fourFoldcross(tr_data, i)[3]))/4
        Q9_results_file.write((str)(mean))
        Q9_results_file.write("\n")
        Q9_results_file.write("Varriance : ")
        varr = statistics.variance(fourFoldcross(tr_data, i))
        Q9_results_file.write((str)(varr))
        Q9_results_file.write("\n")
        Q9_results_file.write("____________________________________________________\n")

    #Makes the CSV file and writes to it
    classify(tr_data, te_data, 12)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["id", "income"])
        for x in te_data:
            writer.writerow((int(x[0]), int(knn_sort(tr_data, x, 12))))

