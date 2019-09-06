import tensorflow as tf


def focal_loss(labels, logits, gamma=2):
    epsilon = 1.e-9
    y_pred = tf.nn.softmax(logits,dim=-1)
    y_pred = y_pred + epsilon # to avoid 0.0 in log
    L = -labels*tf.pow((1-y_pred),gamma)*tf.log(y_pred)
    L = tf.reduce_sum(L)
    batch_size = tf.shape(labels)[0]
    return L / tf.to_float(batch_size)


def get_total_param_num(params, threshold = 1):
    total_parameters = 0
    #iterating over all variables
    for variable in params:
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        if local_parameters >= threshold:
            print("variable {0} with parameter number {1}".format(variable, local_parameters))
        total_parameters+=local_parameters
    print('# total parameter number',total_parameters)
    return total_parameters


def cal_f1(label_num, predicted, truth):
    results = []
    for i in range(label_num):
        results.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    for i, p in enumerate(predicted):
        t = truth[i]
        for j in range(label_num):
            if p[j] == 1:
                if t[j] == 1:
                    results[j]['TP'] += 1
                else:
                    results[j]['FP'] += 1
            else:
                if t[j] == 1:
                    results[j]['FN'] += 1
                else:
                    results[j]['TN'] += 1

    precision = [0.0] * label_num
    recall = [0.0] * label_num
    f1 = [0.0] * label_num
    for i in range(label_num):
        if results[i]['TP'] == 0:
            if results[i]['FP'] == 0 and results[i]['FN'] == 0:
                precision[i] = 1.0
                recall[i] = 1.0
                f1[i] = 1.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0
                f1[i] = 0.0
        else:
            precision[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FP'])
            recall[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FN'])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    # for i in range(label_num):
    #     print(i,results[i], precision[i], recall[i], f1[i])
    return sum(f1) / label_num, sum(precision) / label_num, sum(recall) / label_num
