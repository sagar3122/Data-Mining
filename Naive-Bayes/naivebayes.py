import numpy as np
import pandas as pd
import scipy.stats
import itertools
import matplotlib.pyplot as plt
import sys

'''mean and sd for mv gaussian distribution'''
mean1 = [1,0]
mean2 = [0,1]
sd1 = [[1, 0.75],[0.75, 1]]
sd2 = [[1, 0.75],[0.75, 1]]

'''implementing the differnet parts of the problem given'''
if sys.argv[1] == "2":
    run_code = 1
    samples_generate_list_label_1 = [500]
    samples_generate_list_label_0 = [500]
elif sys.argv[1] == "3":
    run_code = 6
    samples_generate_list_label_1 = [10, 20, 50, 100, 300, 500]
    samples_generate_list_label_0 = [10, 20, 50, 100, 300, 500]
elif sys.argv[1] == "4":
    run_code = 1
    samples_generate_list_label_1 = [300]
    samples_generate_list_label_0 = [700]

# acc_list = []
acc_dict = {}

for x in range(run_code):
    '''training data sets'''
    print("\nsamples used for training data label 0: ",samples_generate_list_label_0[x])
    train_1 = np.random.multivariate_normal(mean1, sd1, samples_generate_list_label_0[x])
    # print(train_1)
    # print(train_1.shape)
    train_df_1 = pd.DataFrame(train_1)
    train_df_1.columns = ["Coordinate_1", "Coordinate_2"]
    # print(train_df_1.columns)
    train_df_1['Class Label'] = 0
    # print(train_df_1)
    print("samples used for training data label 1: ",samples_generate_list_label_1[x])
    train_2 = np.random.multivariate_normal(mean2, sd2, samples_generate_list_label_1[x])
    # print(train_2)
    # print(train_2.shape)
    train_df_2 = pd.DataFrame(train_2)
    train_df_2.columns = ["Coordinate_1", "Coordinate_2"]
    train_df_2['Class Label'] = 1
    # print(train_df_2)

    train_df = train_df_1.append(train_df_2, ignore_index = True, sort = False)
    train_df = train_df.sample(frac=1).reset_index(drop = True)
    # print(train_df)
    '''computing the prior for both the labels'''
    count_label_1 = 0
    count_label_0 = 0
    for index, row in train_df.iterrows():
        if row["Class Label"] == 1:
            count_label_1 = count_label_1 + 1
        else:
            count_label_0 = count_label_0 + 1
    # print("count_label_1: ",count_label_1)
    # print("count_label_0: ",count_label_0)
    prior_label_1 = (count_label_1)/(len(train_df.index))
    print("prior_label_1: ",prior_label_1)
    prior_label_0 = (count_label_0)/(len(train_df.index))
    print("prior_label_0: ",prior_label_0)

    X = train_df[["Coordinate_1", "Coordinate_2"]]
    # print(X)
    Y = train_df[["Class Label"]]
    # print(Y)

    '''testing data sets'''
    test_1 = np.random.multivariate_normal(mean1, sd1, 500)
    # print(test_1)
    # print(test_1.shape)
    test_df_1 = pd.DataFrame(test_1)
    test_df_1.columns = ["Coordinate_1", "Coordinate_2"]
    # print(test_df_1.columns)
    test_df_1['Class Label'] = str(0)
    # print(test_df_1)
    test_2 = np.random.multivariate_normal(mean2, sd2, 500)
    # print(test_2)
    # print(test_2.shape)
    test_df_2 = pd.DataFrame(test_2)
    test_df_2.columns = ["Coordinate_1", "Coordinate_2"]
    test_df_2['Class Label'] = str(1)
    # print(test_df_2)

    test_df = test_df_1.append(test_df_2, ignore_index = True, sort = False)
    test_df = test_df.sample(frac=1).reset_index(drop = True)
    # print(test_df)
    X_Test = test_df
    # print(X_Test)
    Y_Test = pd.DataFrame()
    # print(Y_Test)


    def partition_by_label(dataframe):
        train_set_label_1 = pd.DataFrame()
        train_set_label_0 = pd.DataFrame()
        for index, row in dataframe.iterrows():
            if row["Class Label"] == 1:
                train_set_label_1 = train_set_label_1.append(pd.Series([row["Coordinate_1"], row["Coordinate_2"]]), ignore_index = True)
                # print(train_set_label_1)
            else:
                train_set_label_0 = train_set_label_0.append(pd.Series([row["Coordinate_1"], row["Coordinate_2"]]), ignore_index = True)
                # print(train_set_label_0)
        train_set_label_1.columns = ["Coordinate_1", "Coordinate_2"]
        train_set_label_0.columns = ["Coordinate_1", "Coordinate_2"]
        # print(train_set_label_1)
        # print(train_set_label_0)
        # print(train_set_label_1.dtypes)
        # print(train_set_label_0.dtypes)
        return train_set_label_1, train_set_label_0

    def mean_and_std(dataframe):
        mean = dataframe.mean(axis = 0)
        std = dataframe.std(axis = 0)
        return mean, std

    def attribute_likehood(att, mean, std):
        att_likelihood = scipy.stats.norm(mean, std).pdf(att)
        # print("att_likelihood: ",att_likelihood)
        return att_likelihood

    def label_wise_instance_likelihood(ins, prior, mean, std):
        ins_likelihood = prior
        for att in ins:
            # print(att)
            if att == ins[0]:
                # print("first attribute")
                ins_likelihood *= attribute_likehood(att, mean.loc["Coordinate_1"], std.loc["Coordinate_1"])
            else:
                # print("second attribute")
                ins_likelihood *= attribute_likehood(att, mean.loc["Coordinate_2"], std.loc["Coordinate_2"])
        # print("ins_likelihood: ",ins_likelihood)
        return ins_likelihood



    def myNB(X, Y, X_Test, Y_Test):
        train_set = X.join(Y)
        # print(train_set)
        '''preparing model'''
        train_set_label_1, train_set_label_0 = partition_by_label(train_set)
        # print(train_set_label_1)
        # print(train_set_label_0)
        train_set_label_1_mean, train_set_label_1_std = mean_and_std(train_set_label_1)
        train_set_label_0_mean, train_set_label_0_std = mean_and_std(train_set_label_0)
        # print(train_set_label_1_mean)
        # print(train_set_label_1_std)
        # print(train_set_label_0_mean)
        # print(train_set_label_0_std)
        '''predicting'''
        for index, row in X_Test.iterrows():
            ins = [row["Coordinate_1"], row["Coordinate_2"]]
            # print("instance: ",ins)
            '''for label 1'''
            # print("for label 1")
            label_1_instance_likelihood = label_wise_instance_likelihood(ins, prior_label_1, train_set_label_1_mean, train_set_label_1_std)
            '''for label 0'''
            # print("for label 0")
            label_0_instance_likelihood = label_wise_instance_likelihood(ins, prior_label_0, train_set_label_0_mean, train_set_label_0_std)
            '''normalizing the likelihoods'''
            label_1_instance_probability = label_1_instance_likelihood/(label_1_instance_likelihood + label_0_instance_likelihood)
            label_0_instance_probability = label_0_instance_likelihood/(label_1_instance_likelihood + label_0_instance_likelihood)
            if label_1_instance_probability > label_0_instance_probability:
                label = str(1)
                # print("prediction is label: ",label)
                # posterior = label_1_instance_probability
            else:
                label = str(0)
                # print("prediction is label: ",label)
                # posterior = label_0_instance_probability
            # print("----------------------------------------------------------------------------------------")
            Y_Test = Y_Test.append(pd.Series([row["Coordinate_1"], row["Coordinate_2"], label_1_instance_probability, label_0_instance_probability, label]), ignore_index = True)
        Y_Test.columns = ["Coordinate_1", "Coordinate_2", "label_1_posterior", "label_0_posterior", "predicted_label"]
        print(Y_Test)

        '''testing accuracy'''
        # print(X_Test)
        count_match = 0
        accuracy_df = pd.DataFrame()
        for (index1, row1), (index2, row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
            # print("Yes")
            if row1["predicted_label"] == row2["Class Label"]:
                count_match += 1
                # print("Match found")
                accuracy_df = accuracy_df.append(pd.Series([row1["Coordinate_1"], row1["Coordinate_2"], row1["predicted_label"], row2["Class Label"]]), ignore_index = True)
            else:
                # print("No Match found")
                continue
        accuracy_df.columns = ["Coordinate_1", "Coordinate_2", "predicted_label", "Actual_Class_Label"]
        # print(accuracy_df)
        print("predictions_matched: ",count_match)
        print("Total instances: ",len(Y_Test.index))
        accuracy = (count_match)/(len(Y_Test.index))*100
        print("accuracy: ", accuracy)
        print("error: ",(100-accuracy))


        '''computing precision/recall/confusion matrix'''
        count_TP = 0
        count_TN = 0
        '''TP/TN'''
        for index, row in accuracy_df.iterrows():
            if row["predicted_label"] == "1":
                count_TP += 1
            else:
                count_TN += 1

        '''FP/FN'''
        count_FP = 0
        count_FN = 0
        for (index1, row1), (index2, row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
            if row1["predicted_label"] == "1" and row2["Class Label"] == "0":
                count_FP += 1
            if row1["predicted_label"] == "0" and row2["Class Label"] == "1":
                count_FN += 1
        print("\n---------------------------------------------------------------------")
        print("confusion matrix")
        print("---------------------------------------------------------------------")
        confusion_matrix_dict = {('actual class','positive'):[count_TP,count_FN], ('actual class', 'negative'):[count_FP,count_TN]}
        index = [['predicted label','predicted label'],['positive', 'negative']]
        confusion_matrix = pd.DataFrame(confusion_matrix_dict, index = index)
        print(confusion_matrix)
        print("---------------------------------------------------------------------")
        # print("TP: ",count_TP)
        # print("FP: ",count_FP)
        # print("TN: ",count_TN)
        # print("FN: ",count_FN)
        # print("TP + FP: ", count_TP + count_FP)
        precision = count_TP/(count_TP + count_FP)
        print("\nprecision: ",precision*100)
        # print("TP + FN: ", count_TP + count_FN)
        recall = count_TP/(count_TP + count_FN)
        print("recall: ",recall*100)

        '''scatter plot'''
        try:
            if sys.argv[1] == "2" and sys.argv[2] == "Scatter":
                for index, row in accuracy_df.iterrows():
                    if row["predicted_label"] == "1":
                        plt.scatter(row["Coordinate_1"], row["Coordinate_2"], color = "red")
                    else:
                        plt.scatter(row["Coordinate_1"], row["Coordinate_2"], color = "green")

                plt.xlabel("xplots")
                plt.ylabel("yplots")
                plt.show()
        except (IndexError):
            pass



        '''store accuracy values'''
        # acc_list.append(accuracy)
        acc_dict[samples_generate_list_label_1[x]] = accuracy

        '''ROC PLOT'''
        try:
            if (sys.argv[1] == "2" or sys.argv[1] == "4") and sys.argv[2] == "ROC":
                ROC_df = pd.DataFrame()
                for (index1, row1), (index2, row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
                    ROC_df = ROC_df.append(pd.Series([row1["Coordinate_1"], row1["Coordinate_2"], row2["Class Label"], row1["label_1_posterior"], row1["label_0_posterior"]]), ignore_index = True)
                ROC_df.columns = ["Coordinate_1", "Coordinate_2", "Actual_Class_Label", "Posterior_label_1", "Posterior_label_0"]
                # print(ROC_df)
                ROC_df = ROC_df.sort_values(by = "Posterior_label_1", ascending = False)
                # print(ROC_df)
                '''setting thresholds--making predictions--computing the TP and FP rates'''
                ROC_df["Predicted_Label"] = str(0)
                # print(ROC_df)
                ROC_rates_list = []
                ROC_rates_list.append((0.0,0.0)) #when the threshold is 1
                for index, row in ROC_df.iterrows():
                    threshold = row["Posterior_label_1"]
                    # print("threshold: ",threshold)
                    for index1, row1 in ROC_df.iterrows():
                        if float(row1["Posterior_label_1"]) >= float(threshold):
                            # print("True")
                            ROC_df.at[index1, 'Predicted_Label'] = str(1)
                            # break
                        else:
                            # print("False")
                            continue
                    # break
                # print(ROC_df)
                    TP_count = 0
                    TN_count = 0
                    FP_count = 0
                    FN_count = 0
                    for index2, row2 in ROC_df.iterrows():
                        if row2["Predicted_Label"] == "1" and row2["Actual_Class_Label"] == "1":
                            # print("Inc TP")
                            TP_count += 1
                        elif row2["Predicted_Label"] == "1" and row2["Actual_Class_Label"] == "0":
                            # print("Inc FP")
                            FP_count += 1
                        elif row2["Predicted_Label"] == "0" and row2["Actual_Class_Label"] == "0":
                            # print("Inc TN")
                            TN_count += 1
                        elif row2["Predicted_Label"] == "0" and row2["Actual_Class_Label"] == "1":
                            # print("Inc FN")
                            FN_count += 1
                    # print("TP: ",TP_count)
                    # print("FP: ",FP_count)
                    # print("TN: ",TN_count)
                    # print("FN: ",FN_count)
                    TPR = TP_count/(TP_count + FN_count)
                    # print("Recall: ",TPR)
                    FPR = FP_count/(TN_count + FP_count)
                    # print("FPR: ",FPR)
                    ROC_rates_list.append((FPR, TPR))
                # print(ROC_rates_list)
                '''plot the ROC curve'''
                h,k = zip(*ROC_rates_list)
                plt.plot(h,k)
                # for items in ROC_rates_list:
                #         plt.plot(items[0],items[1])
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.show()
                '''AUC'''
                AUC_list = []
                # print("ROC_rates_list: ",ROC_rates_list)
                ROC_rates_list_1 = list(ROC_rates_list)
                ROC_rates_list_1.remove((1.0,1.0))
                # print("ROC_rates_list_1: ",ROC_rates_list_1)
                # print("ROC_rates_list_1_len: ",len(ROC_rates_list_1))
                ROC_rates_list_2 = list(ROC_rates_list)
                ROC_rates_list_2.remove((0.0,0.0))
                # print("ROC_rates_list_2: ",ROC_rates_list_2)
                # print("ROC_rates_list_2_len: ",len(ROC_rates_list_2))
                for n, m in zip(ROC_rates_list_1, ROC_rates_list_2):
                    # print("m: ",m)
                    # print("m[0]: ",m[0])
                    # print("m[1]: ",m[1])
                    # print("n:",n)
                    # print("n[0]: ",n[0])
                    # print("n[1]: ",n[1])
                    breadth = m[0] - n[0]
                    length = m[1] - 0
                    rectangle_area = length * breadth
                    # print("rectangle_area: ",rectangle_area)
                    AUC_list.append(rectangle_area)
                # print("AUC_list: ",AUC_list)

                AUC = 0.0
                for a in AUC_list:
                    AUC = AUC + float(a)
                print("AUC: ",AUC)

        except (IndexError):
            pass


    myNB(X,Y, X_Test, Y_Test)

'''part 3'''
if sys.argv[1] == "3":
    # print("accuracy list: ", acc_list)
    print("accuracy dictionary: ",acc_dict)
    # for key, values in acc_dict.items():
    #     # print(key)
    #     # print(type(key))
    #     # print(values)
    #     # print(type(values))
    #     plt.plot(key, values, data='object')
    lists = sorted(acc_dict.items())
    x,y = zip(*lists)
    plt.plot(x,y)
    plt.xlabel("no of samples drawn")
    plt.ylabel("corresponding testing accuracies")
    plt.show()
