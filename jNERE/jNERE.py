# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time
from utils import readCoNLL2004_prepared_corpus, vocabNER, readData, getRelVocab, convert2BIO
from BIOF1Validation import compute_NER_f1_macro, compute_Rel_f1_macro, getCorrectNERids
import logging

if __name__ == '__main__':
        parser = OptionParser()
        #parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
        #parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
        #parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
        parser.add_option("--output", dest="output", help="File name for predicted output", metavar="FILE", default="outputs/output.pred")
        parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
        parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
        #parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
        parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=50)
        parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=25)
        parser.add_option("--nembedding", type="int", dest="nembedding_dims", default=50)
        parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=50)
        parser.add_option("--epochs", type="int", dest="epochs", default=50)
        parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
        parser.add_option("--mixture", type="float", dest="mixture_weight", default=0.5)
        parser.add_option("--lr", type="float", dest="learning_rate", default=None)
        #parser.add_option("--outdir", type="string", dest="output", default="")
        parser.add_option("--activation", type="string", dest="activation", default="tanh")
        parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
        parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=100)
        parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
        parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
        parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
        parser.add_option("--bibi-lstm", action="store_false", dest="bibiFlag", default=True)
        parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
        parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
        parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
        #parser.add_option("--dynet-weight-decay", type="float", dest="wdc", default=0.0001)

        (options, args) = parser.parse_args()

        highestScore = 0.0
        eId = 0

        train_id2sent, train_id2pos, train_id2ner, train_id2nerBILOU, train_id2arg2rel, test_id2sent, test_id2pos, test_id2ner, test_id2nerBILOU, test_id2arg2rel = readCoNLL2004_prepared_corpus()
        words, w2i, c2i, nertags, postagCount = vocabNER(train_id2sent, train_id2pos, train_id2nerBILOU)

        id2arg2rel, rels, classweights = getRelVocab(train_id2arg2rel, train_id2nerBILOU)

        fulltrain_data = readData(train_id2sent, train_id2pos, c2i)
        test_data = readData(test_id2sent, test_id2pos, c2i)


        #print w2i
        #print c2i
        #print nertags
        #print postags

        train_data, train_id2arg2rel_train = {}, {}
        numInstances = len(fulltrain_data) / 5 * 4
        count = 0
        for index in fulltrain_data:
            train_data[index] = fulltrain_data[index]
            train_id2arg2rel_train[index]= train_id2arg2rel[index]

            count += 1
            if count >= numInstances:
                break

        dev_data = {}
        dev_id2arg2rel = {}
        for index in fulltrain_data:
            if index not in train_data:
                dev_data[index] = fulltrain_data[index]
                dev_id2arg2rel[index] = train_id2arg2rel[index]


        #parser = learner.jNeRE(words, nertags, postagCount, rels, w2i, c2i, options)
        parser = learner.jNeRE(words, nertags, rels, w2i, c2i, options)

        for epoch in xrange(options.epochs):
            print '\n-----------------\nStarting epoch', epoch + 1

            #parser.Train(train_data, train_id2nerBILOU, id2arg2rel, classweights)
            parser.Train(train_data, train_id2nerBILOU, id2arg2rel)

            label_pred = []
            label_correct = []
            predDev, relsDev = parser.Predict(dev_data)
            #pickle.dump((predDev, relsDev), open(options.output + "dev_ep" + str(epoch + 1), "wb"))

            for sentenceID in predDev:
                label_pred.append(predDev[sentenceID])
                label_correct.append(train_id2nerBILOU[sentenceID].strip().split())

            assert len(label_pred) == len(label_correct)

            f1 = compute_NER_f1_macro(label_pred, label_correct, 'O', "IOBES")
            f1_b = compute_NER_f1_macro(label_pred, label_correct, 'B', "IOBES")

            if f1_b > f1:
                logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
                f1 = f1_b

            print "NER macro-averaged F1 on dev: {:.3f}".format(f1)

            correctNerIDs = getCorrectNERids(predDev, train_id2nerBILOU)
            f1_rel = compute_Rel_f1_macro(relsDev, dev_id2arg2rel, correctNerIDs)
            print "RC macro-averaged F1 on dev: {:.3f}".format(f1_rel)

            score = (f1 + f1_rel) / 2
            print "(NER-F1 + RC-F1)/2 on dev:", score
            if score >= highestScore:
                #parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                highestScore = score
                eId = epoch + 1

            print "Highest (NER-F1 + RC-F1)/2 on dev: %.3f at epoch %d" % (highestScore, eId)

            label_pred = []
            label_correct = []
            predDev, relsDev = parser.Predict(test_data)
            #pickle.dump((predDev, relsDev), open(options.output + "test_ep" + str(epoch + 1), "wb"))
            for sentenceID in predDev:
                label_pred.append(predDev[sentenceID])
                label_correct.append(test_id2nerBILOU[sentenceID].strip().split())

            assert len(label_pred) == len(label_correct)

            f1 = compute_NER_f1_macro(label_pred, label_correct, 'O', "IOBES")
            f1_b = compute_NER_f1_macro(label_pred, label_correct, 'B', "IOBES")

            if f1_b > f1:
                logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
                f1 = f1_b

            print "NER macro-averaged F1 on set: {:.3f}".format(f1)

            correctNerIDs = getCorrectNERids(predDev, test_id2nerBILOU)
            f1_rel = compute_Rel_f1_macro(relsDev, test_id2arg2rel, correctNerIDs)
            print "RC macro-averaged F1 on test: {:.3f}".format(f1_rel)

