import json
import time

import numpy as np

from dataset import DataSet,load_data_from_csv,seg_words
from model_cnn import Model_CNN
from utils import *
import os,argparse


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # mode
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")

    # # data
    # parser.add_argument("--data_files", type=str, nargs='+', default=None, help="data file for train or inference")
    # parser.add_argument("--eval_files", type=str, nargs='+', default=None, help="eval data file for evaluation")
    # parser.add_argument("--label_file", type=str, default=None, help="label file")
    # parser.add_argument("--vocab_file", type=str, default=None, help="vocab file")
    # parser.add_argument("--embed_file", type=str, default=None, help="embedding file to restore")
    # parser.add_argument("--out_file", type=str, default=None, help="output file for inference")
    # parser.add_argument("--split_word", type='bool', nargs="?", const=True, default=True,
    #                     help="Whether to split word when oov")
    # parser.add_argument("--max_len", type=int, default=1200, help='max length for doc')
    # parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # parser.add_argument("--reverse", type='bool', nargs="?", const=True, default=False, help="Whether to reverse data")
    # parser.add_argument("--prob", type='bool', nargs="?", const=True, default=False, help="Whether to export prob")

    # model
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--decay_schema", type=str, default='hand', help='learning rate decay: exp | hand')
    parser.add_argument("--encoder", type=str, default='gnmt', help="gnmt | elmo")
    parser.add_argument("--decay_steps", type=int, default=10000, help="decay steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. RMS: 0.001 | 0.0001")
    parser.add_argument("--focal_loss", type=float, default=2., help="gamma of focal loss")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="embedding_dropout")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="drop out keep ratio for training")
    parser.add_argument("--weight_keep_drop", type=float, default=0.8, help="weight keep drop")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.0, help="l2 loss lambda")
    parser.add_argument("--rnn_cell_name", type=str, default='lstm', help='rnn cell name')
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding_size")
    parser.add_argument("--num_units", type=int, default=384, help="num_units")
    parser.add_argument("--double_decoder", type='bool', nargs="?", const=True, default=False,
                        help="Whether to double decoder size")
    parser.add_argument("--variational_dropout", type='bool', nargs="?", const=True, default=True,
                        help="Whether to use variational_dropout")
    parser.add_argument("--filter_sizes", type=str,default="3,4,5",)
    parser.add_argument("--num_filters", type=int,default=128, )

    # clf
    parser.add_argument("--target_label_num", type=int, default=4, help="target_label_num")
    parser.add_argument("--feature_num", type=int, default=20, help="feature_num")

    # train
    parser.add_argument("--need_early_stop", type='bool', nargs="?", const=True, default=True,
                        help="Whether to early stop")
    parser.add_argument("--patient", type=int, default=5, help="patient of early stop")
    parser.add_argument("--debug", type='bool', nargs="?", const=True, default=False, help="Whether use debug mode")
    parser.add_argument("--num_train_epoch", type=int, default=50, help="training epoches")
    parser.add_argument("--steps_per_stats", type=int, default=20, help="steps to print stats")
    parser.add_argument("--steps_per_summary", type=int, default=50, help="steps to save summary")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="steps to save model")

    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/visual-semantic',
                        help="checkpoint dir to save model")


def convert_to_hparams(params):
    hparams = tf.contrib.training.HParams()
    for k, v in params.items():
        hparams.add_hparam(k, v)
    return hparams


def inference(flags):
    print("inference data file {0}".format(flags.data_files))
    dataset = DataSet(flags.data_files, flags.label_file, flags.batch_size, reverse=flags.reverse,
                      split_word=flags.split_word, max_len=flags.max_len)
    hparams = {} # todo
    with tf.Session() as sess:
        model = Model_CNN(hparams)
        model.build()

        try:
            model.restore_model(sess)  # restore best solution
        except Exception as e:
            print("unable to restore model with exception", e)
            exit(1)

        scalars = model.scalars.eval(session=sess)
        print("Scalars:", scalars)
        weight = model.weight.eval(session=sess)
        print("Weight:", weight)
        cnt = 0
        for (source, lengths, _, ids) in dataset.get_next(shuffle=False):
            predict, logits = model.inference_clf_one_batch(sess, source, lengths)
            for i, (p, l) in enumerate(zip(predict, logits)):
                for j in range(flags.feature_num):
                    label_name = dataset.i2l[j]
                    if flags.prob:
                        tag = [float(v) for v in l[j]]
                    else:
                        tag = dataset.tag_i2l[np.argmax(p[j])]
                    dataset.items[cnt + i][label_name] = tag
            cnt += len(lengths)
            print("\r# process {0:.2%}".format(cnt / dataset.data_size), new_line=False)

    print("# Write result to file ...")
    with open(flags.out_file, 'w') as f:
        for item in dataset.items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("# Done")


def train_eval_clf(model, sess, dataset):
    from collections import defaultdict
    checkpoint_loss, acc = 0.0, 0.0

    predicts, truths = defaultdict(list), defaultdict(list)
    for i, (source, lengths, targets, _) in enumerate(dataset.get_next(shuffle=False)):
        batch_loss, accuracy, batch_size, predict = model.eval_clf_one_step(sess, source, lengths, targets)
        # batch * 20 * 4
        for i, p in enumerate(predict):
            for j in range(model.hparams.feature_num):
                label_name = dataset.i2l[j]
                truths[label_name].append(targets[i][j])
                predicts[label_name].append(p[j])
        checkpoint_loss += batch_loss
        acc += accuracy
        if (i + 1) % 100 == 0:
            print("# batch %d/%d" % (i + 1, dataset.num_batches))

    results = {}
    total_f1 = 0.0
    for label_name in dataset.label_names:
        # print("# Get f1 score for",label_name)
        f1, precision, recall = cal_f1(model.hparams.target_label_num, np.asarray(predicts[label_name]),
                                       np.asarray(truths[label_name]))
        results[label_name] = f1
        total_f1 += f1
        print("# {0} - {1}".format(label_name, f1))

    final_f1 = total_f1 / len(results)

    print("# Eval loss %.5f, f1 %.5f" % (checkpoint_loss / i, final_f1))
    return -1 * final_f1, checkpoint_loss / i


def train_clf(data_info,flags):
    dataset = DataSet(datas=data_info["train_data"], labels=data_info["train_label"], batch_size=data_info["batch_size"], reverse=data_info["reverse"],
                      split_word=data_info["split_word"], max_len=data_info["max_len"])
    eval_dataset = DataSet(data_info["eval_data"], data_info["eval_label"], batch_size=data_info["batch_size"], reverse=data_info["reverse"],
                      split_word=data_info["split_word"], max_len=data_info["max_len"])

    params = vars(flags)
    hparams = convert_to_hparams(params)
    hparams.add_hparam("vocab_size",10913)
    hparams.add_hparam("seq_len",data_info["max_len"])
    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    with train_graph.as_default():
        train_model = Model_CNN(hparams)
        train_model.build()
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        # eval_hparams = {}  # todo
        eval_model = Model_CNN(hparams)
        eval_model.build()

    train_sess = tf.Session(graph=train_graph)
    train_model.init_model(train_sess, initializer=initializer)
    try:
        train_model.restore_model(train_sess)
    except:
        print("unable to restore model, train from scratch")

    print("# Start to train with learning rate {0}, {1}".format(flags.learning_rate, time.ctime()))

    global_step = train_sess.run(train_model.global_step)
    print("# Global step", global_step)

    eval_ppls = []
    best_eval = 1000000000
    pre_best_checkpoint = None
    final_learn = 2
    for epoch in range(flags.num_train_epoch):
        step_time, checkpoint_loss, acc, iters = 0.0, 0.0, 0.0, 0
        for i, (source, lengths, targets, _) in enumerate(dataset.get_next()):
            start_time = time.time()
            add_summary = (global_step % flags.steps_per_summary == 0)
            batch_loss, global_step, accuracy, token_num, batch_size = \
                train_model.train_clf_one_step(train_sess,
                                              source,
                                              lengths, targets,
                                              add_summary=add_summary,
                                              run_info=add_summary and flags.debug)
            step_time += (time.time() - start_time)
            checkpoint_loss += batch_loss
            acc += accuracy
            iters += token_num

            if global_step == 0:
                continue

            if global_step % flags.steps_per_stats == 0:
                train_acc = (acc / flags.steps_per_stats) * 100
                acc_summary = tf.Summary()
                acc_summary.value.add(tag='accuracy', simple_value=train_acc)
                train_model.summary_writer.add_summary(acc_summary, global_step=global_step)

                print(
                    "# Epoch %d  global step %d loss %.5f batch %d/%d lr %g "
                    "accuracy %.5f wps %.2f step time %.2fs" %
                    (epoch + 1, global_step, checkpoint_loss / flags.steps_per_stats, i + 1, dataset.num_batches,
                     train_model.learning_rate.eval(session=train_sess),
                     train_acc, (iters) / step_time, step_time / (flags.steps_per_stats)))
                step_time, checkpoint_loss, iters, acc = 0.0, 0.0, 0, 0.0

            if global_step % flags.steps_per_eval == 0:
                print("# global step {0}, eval model at {1}".format(global_step, time.ctime()))
                checkpoint_path = train_model.save_model(train_sess)
                with tf.Session(graph=eval_graph) as eval_sess:
                    eval_model.init_model(eval_sess)
                    eval_model.restore_ema_model(eval_sess, checkpoint_path)
                    eval_ppl, eval_loss = train_eval_clf(eval_model, eval_sess, eval_dataset)
                    print("# current result {0}, previous best result {1}".format(eval_ppl, best_eval))
                    loss_summary = tf.Summary()
                    loss_summary.value.add(tag='eval_loss', simple_value=eval_loss)
                    train_model.summary_writer.add_summary(loss_summary, global_step=global_step)
                    if eval_ppl < best_eval:
                        pre_best_checkpoint = checkpoint_path
                        eval_model.save_model(eval_sess, global_step)
                        best_eval = eval_ppl
                    eval_ppls.append(eval_ppl)
        print("# Finsh epoch {1}, global step {0}".format(global_step, epoch + 1))
    print("# Best accuracy {0}".format(best_eval))


def main():
    train_data_path = "/Users/withheart/Documents/studys/senmantic/data/data_sample/sentiment_analysis_trainingset_sample.csv"
    validate_data_path = "/Users/withheart/Documents/studys/senmantic/data/data_sample/sentiment_analysis_validationset_sample.csv"
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    print("start load data")
    train_data_df = load_data_from_csv(train_data_path)
    validate_data_df = load_data_from_csv(validate_data_path)

    content_train = train_data_df.iloc[:, 1]
    content_validate = validate_data_df.iloc[:, 1]

    print("start seg train data")
    content_train = seg_words(content_train)
    content_validate = seg_words(content_validate)
    print("complete seg train data")

    columns = train_data_df.columns.values.tolist()

    label_train = train_data_df.iloc[:,2].values
    label_validate = validate_data_df.iloc[:,2].values

    # model train
    print("start train and validate model")
    data_info = {}
    data_info["train_data"] = content_train
    data_info["train_label"] = label_train
    data_info["batch_size"] = 64
    data_info["reverse"] = True
    data_info["split_word"] = False
    data_info["max_len"] = max([len(i) for i in content_train])
    data_info["eval_data"] = content_validate
    data_info["eval_label"] = label_validate
    data_info["num_train_epoch"] = 10

    train_clf(data_info,flags)
    print("complete train and validate model")


if __name__ == "__main__":
    main()
