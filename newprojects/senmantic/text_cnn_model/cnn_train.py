#!/user/bin/env python
import tensorflow as tf
from data_process import load_data_from_csv
import config
import logging
from text_cnn_model.text_cnn import TextCNN
from tensorflow.contrib import learn
import numpy as np
import pandas as pd

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

FLAGS = tf.flags.FLAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-mn', '--model_name', type=str, nargs='?',
    #                     help='the name of model')
    #
    # args = parser.parse_args()
    # model_name = args.model_name
    # if not model_name:
    #     model_name = "model_dict.pkl"

    # load train data
    logger.info("start load data")
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    content_train = train_data_df.iloc[:, 1]
    content_validate = validate_data_df.iloc[:,1]

    # vocab 大小
    max_document_length = max([len(x.split(" ")) for x in content_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    content_train_numeric = np.array(list(vocab_processor.fit_transform(content_train)))
    content_validate_numeric = np.array(list(vocab_processor.fit_transform(content_validate)))

    logger.info("start seg train data")
    logger.info("complete seg train data")

    columns = train_data_df.columns.values.tolist()

    # model train
    logger.info("start train model")
    classifier_dict = dict()
    for column in columns[2:]:
        label_train = train_data_df[column]
        label_train_numeric = pd.get_dummies(label_train)[[-2, -1, 0, 1]].values
        cnn = TextCNN()
        cnn.model(
            sequence_length=content_train_numeric.shape[1],
            num_classes=label_train_numeric.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            model_name=column)
        logger.info("start train %s model" % column)
        validate_label = validate_data_df[column]
        validate_label_numeric = pd.get_dummies(validate_label)[[-2, -1, 0, 1]].values
        cnn.cnn_train(content_train_numeric, label_train_numeric,
                      content_validate_numeric,validate_label_numeric,
                      vocab_processor,column)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = cnn

    logger.info("complete train model")

    # validate model
    # content_validate = validate_data_df.iloc[:, 1]
    #
    # logger.info("start seg validate data")
    # logger.info("complete seg validate data")
    #
    # logger.info("start validate model")
    # f1_score_dict = dict()
    # for column in columns[2:]:
    #     label_validate = validate_data_df[column]
    #     text_classifier = classifier_dict[column]
    #     f1_score = text_classifier.get_f1_score(content_validate, label_validate)
    #     f1_score_dict[column] = f1_score
    #
    # f1_score = np.mean(list(f1_score_dict.values()))
    # str_score = "\n"
    # for column in columns[2:]:
    #     str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"
    #
    # logger.info("f1_scores: %s\n" % str_score)
    # logger.info("f1_score: %s" % f1_score)
    # logger.info("complete validate model")
    #
    # # save model
    # logger.info("start save model")
    # model_save_path = config.model_save_path
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    #
    # model_name = "test" # 测试时使用
    # joblib.dump(classifier_dict, model_save_path + model_name)
    # logger.info("complete save model")