# coding=utf-8
from dynet import *
import dynet
from utils import load_embeddings_file, orthonormal_initializer
from operator import itemgetter
import utils, time, random, crf
import numpy as np
from mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor
from crf import CRF


class jEcRE:
    def __init__(self, vocab, ner, postagCount, rels, w2i, c2i, options):
        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)
        if options.learning_rate is not None:
            self.trainer = AdamTrainer(self.model, alpha=options.learning_rate)
            print("Adam initial learning rate:", options.learning_rate)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.ner = {word: ind for ind, word in enumerate(ner)}
        self.id2ner = {ind: word for ind, word in enumerate(ner)}
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.id2rels = rels
        # print self.rels
        # print self.id2rels
        self.nerdims = options.nembedding_dims
        self.mixture_weight = options.mixture_weight
        self.posCount = postagCount

        self.pos2id = {word: ind + 1 for ind, word in enumerate(postagCount.keys())}
        self.pdims = options.pembedding_dims

        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        self.nerlookup = self.model.add_lookup_parameters((len(ner), self.nerdims))
        self.plookup = self.model.add_lookup_parameters((len(postagCount.keys()) + 1, self.pdims))

        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = load_embeddings_file(options.external_embedding, lower=True)
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.vocab:
                _word = unicode(word, "utf-8")
                if _word in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.vocab[word], ext_embeddings[_word])
            print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.vocab), count))

        self.ner_builders = [VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model)]
        self.ner_bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]

        if self.bibiFlag:
            self.builders = [
                VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims, self.model),
                VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [
                VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims,
                                   self.model),
                VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims,
                                   self.model)]
        else:
            self.builders = [
                SimpleRNNBuilder(1, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims, self.model),
                SimpleRNNBuilder(1, self.wdims + self.cdims * 2 + self.nerdims + self.pdims, self.ldims, self.model)]

        # self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims * 2, len(self.ner), softmax))

        self.hidden_units = options.hidden_units

        self.char_rnn = RNNSequencePredictor(LSTMBuilder(1, self.cdims, self.cdims, self.model))

        self.crf_module = CRF(self.model, self.id2ner)

        self.tanh_layer_W = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
        self.tanh_layer_b = self.model.add_parameters((self.hidden_units))

        self.last_layer_W = self.model.add_parameters((len(self.ner), self.hidden_units))
        self.last_layer_b = self.model.add_parameters((len(self.ner)))

        W = orthonormal_initializer(self.hidden_units, 2 * self.ldims)

        self.head_layer_W = self.model.parameters_from_numpy(W)
        self.head_layer_b = self.model.add_parameters((self.hidden_units,),
                                               init=dynet.ConstInitializer(0.))

        self.dep_layer_W = self.model.parameters_from_numpy(W)
        self.dep_layer_b = self.model.add_parameters((self.hidden_units,),
                                               init=dynet.ConstInitializer(0.))

        self.rel_U = self.model.add_parameters((len(self.rels) * self.hidden_units, self.hidden_units),
                                               init=dynet.ConstInitializer(0.))

        self.rel_W = self.model.parameters_from_numpy(orthonormal_initializer(len(self.rels), 2 * self.hidden_units))
        #self.rel_W = self.model.add_parameters((len(self.rels), self.hidden_units * 2))
        self.rel_b = self.model.add_parameters((len(self.rels),),
                                               init=dynet.ConstInitializer(0.))

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def Predict(self, test_data):
        # with open(conll_path, 'r') as conllFP:
        outputPredNER = {}
        id2arg2rel = {}
        outputPredRel = {}
        count = 0.0
        nercount = 0.0
        for sentenceID in test_data:
            sentence = test_data[sentenceID]

            for entry in sentence:
                wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                posvec = self.plookup[int(self.pos2id.get(entry.pos, 0))]
                last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                    -1]

                entry.vec = concatenate(filter(None, [wordvec, posvec, last_state, rev_last_state]))

                entry.ner_lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            # Predicted ner tags
            lstm_forward = self.ner_builders[0].initial_state()
            lstm_backward = self.ner_builders[1].initial_state()
            for entry, rentry in zip(sentence, reversed(sentence)):
                lstm_forward = lstm_forward.add_input(entry.vec)
                lstm_backward = lstm_backward.add_input(rentry.vec)

                entry.ner_lstms[1] = lstm_forward.output()
                rentry.ner_lstms[0] = lstm_backward.output()

            for entry in sentence:
                entry.ner_vec = concatenate(entry.ner_lstms)

            blstm_forward = self.ner_bbuilders[0].initial_state()
            blstm_backward = self.ner_bbuilders[1].initial_state()

            for entry, rentry in zip(sentence, reversed(sentence)):
                blstm_forward = blstm_forward.add_input(entry.ner_vec)
                blstm_backward = blstm_backward.add_input(rentry.ner_vec)
                entry.ner_lstms[1] = blstm_forward.output()
                rentry.ner_lstms[0] = blstm_backward.output()

            concat_layer = [concatenate(entry.ner_lstms) for entry in sentence]

            context_representations = [dynet.tanh(dynet.affine_transform([self.tanh_layer_b.expr(),
                                                                          self.tanh_layer_W.expr(),
                                                                          context])) \
                                       for context in concat_layer]

            tag_scores = [dynet.affine_transform([self.last_layer_b.expr(),
                                                  self.last_layer_W.expr(),
                                                  context]) \
                          for context in context_representations]

            observations = [dynet.concatenate([obs, dynet.inputVector([-1e10, -1e10])], d=0) for obs in
                            tag_scores]

            predicted_ner_indices, _ = self.crf_module.viterbi_decoding(observations)

            predicted_nertags = [self.id2ner[idx] for idx in predicted_ner_indices]

            for ind in range(len(predicted_nertags)):
                if sentence[ind].pos != "O":
                    predicted_nertags[ind] = sentence[ind].pos + "-" + predicted_nertags[ind]

            outputPredNER[sentenceID] = predicted_nertags

            # Add ner embeddings
            for entry, ner in zip(sentence, predicted_ner_indices):
                entry.vec = concatenate([entry.vec, self.nerlookup[ner]])
                entry.lstms = [entry.vec, entry.vec]

            # Relation losses
            if self.blstmFlag:
                lstm_forward = self.builders[0].initial_state()
                lstm_backward = self.builders[1].initial_state()

                for entry, rentry in zip(sentence, reversed(sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = lstm_forward.output()
                    rentry.lstms[0] = lstm_backward.output()

                if self.bibiFlag:
                    for entry in sentence:
                        entry.vec = concatenate(entry.lstms)

                    blstm_forward = self.bbuilders[0].initial_state()
                    blstm_backward = self.bbuilders[1].initial_state()

                    for entry, rentry in zip(sentence, reversed(sentence)):
                        blstm_forward = blstm_forward.add_input(entry.vec)
                        blstm_backward = blstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = blstm_forward.output()
                        rentry.lstms[0] = blstm_backward.output()

            concat_layer = [concatenate(entry.lstms) for entry in sentence]

            head_context_representations = [dynet.tanh(dynet.affine_transform([self.head_layer_b.expr(),
                                                                               self.head_layer_W.expr(),
                                                                               context])) \
                                            for context in concat_layer]

            dep_context_representations = [dynet.tanh(dynet.affine_transform([self.dep_layer_b.expr(),
                                                                              self.dep_layer_W.expr(),
                                                                              context])) \
                                           for context in concat_layer]

            nerids = []
            for ind, tag in enumerate(predicted_nertags):
                # print(ind, tag)
                if str(tag).startswith("L-") or str(tag).startswith("U-"):
                    if not str(tag).endswith("-Other"):
                        nerids.append(ind)

            id2arg2rel[sentenceID] = {}
            for ind1 in nerids:
                for ind2 in nerids:
                    if ind1 != ind2:
                        if predicted_nertags[ind1] in ["L-Peop", "U-Peop"] and predicted_nertags[ind2] in ["L-Peop",
                                                                                                           "U-Peop",
                                                                                                           "L-Org",
                                                                                                           "U-Org",
                                                                                                           "L-Loc",
                                                                                                           "U-Loc"]:
                            id2arg2rel[sentenceID][(ind1, ind2)] = "NEG"
                        if predicted_nertags[ind1] in ["L-Loc", "U-Loc", "L-Org", "U-Org"] and predicted_nertags[
                            ind2] in ["L-Loc", "U-Loc"]:
                            id2arg2rel[sentenceID][(ind1, ind2)] = "NEG"
                        # id2arg2rel[sentenceID][(ind1, ind2)] = "NEG"

            for (head, dep) in id2arg2rel[sentenceID]:
                # print (head, dep), pairrels[(head, dep)]
                linear = self.rel_U.expr() * dep_context_representations[dep]
                linear = dynet.reshape(linear, (self.hidden_units, len(self.rels)))
                bilinear = dynet.transpose(head_context_representations[head]) * linear
                biaffine = dynet.transpose(bilinear) + self.rel_W.expr() * concatenate(
                    [head_context_representations[head], dep_context_representations[dep]]) + self.rel_b.expr()
                id2arg2rel[sentenceID][(head, dep)] = self.id2rels[np.argmax(softmax(biaffine).value())]

            outputPredRel[sentenceID] = {}
            for (head, dep) in id2arg2rel[sentenceID]:
                rel = id2arg2rel[sentenceID][(head, dep)]
                if rel != "NEG":
                    outputPredRel[sentenceID][(head, dep)] = rel
                # else:
                #    i_rel = id2arg2rel[sentenceID][(dep, head)]
                #    if str(i_rel).endswith("-1"):
                #        outputPredRel[sentenceID][(head, dep)] = i_rel[:-2]

            renew_cg()
            # print "----"
            # print outputPredNER[sentenceID]
            # print id2arg2rel[sentenceID]
            # print outputPredRel[sentenceID]

        return outputPredNER, outputPredRel

    def Train(self, train_data, train_id2nerBILOU, id2arg2rel, isTrain=True):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()
        nwtotal = 0

        if isTrain:
            shuffledData = train_data.keys()
            random.shuffle(shuffledData)

            # errs = []
            lerrs = []
            nerErrs = []

            for iSentence, sentenceId in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print "Processing sentence number: %d" % iSentence, ", Loss: %.4f" % (
                            eloss / etotal), ", Time: %.2f" % (time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                sentence = train_data[sentenceId]
                goldNers = train_id2nerBILOU[sentenceId].strip().split()

                for entry in sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c / (0.25 + c)))
                    wordvec = self.wlookup[
                        int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None

                    #c = float(self.posCount.get(entry.pos, 0))
                    #dropFlag = (random.random() < (c / (0.25 + c)))
                    #posvec = self.plookup[int(self.pos2id.get(entry.pos, 0)) if dropFlag else 0]
                    posvec = self.plookup[int(self.pos2id.get(entry.pos, 0))]

                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                        -1]

                    entry.vec = dynet.dropout(concatenate(filter(None, [wordvec, posvec, last_state, rev_last_state])),
                                              0.33)

                    entry.ner_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                # ner tagging loss
                lstm_forward = self.ner_builders[0].initial_state()
                lstm_backward = self.ner_builders[1].initial_state()
                for entry, rentry in zip(sentence, reversed(sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.ner_lstms[1] = lstm_forward.output()
                    rentry.ner_lstms[0] = lstm_backward.output()

                for entry in sentence:
                    entry.ner_vec = concatenate(entry.ner_lstms)

                blstm_forward = self.ner_bbuilders[0].initial_state()
                blstm_backward = self.ner_bbuilders[1].initial_state()

                for entry, rentry in zip(sentence, reversed(sentence)):
                    blstm_forward = blstm_forward.add_input(entry.ner_vec)
                    blstm_backward = blstm_backward.add_input(rentry.ner_vec)
                    entry.ner_lstms[1] = blstm_forward.output()
                    rentry.ner_lstms[0] = blstm_backward.output()

                concat_layer = [dynet.dropout(concatenate(entry.ner_lstms), 0.33) for entry in sentence]

                context_representations = [dynet.tanh(dynet.affine_transform([self.tanh_layer_b.expr(),
                                                                              self.tanh_layer_W.expr(),
                                                                              context])) \
                                           for context in concat_layer]

                tag_scores = [dynet.affine_transform([self.last_layer_b.expr(),
                                                      self.last_layer_W.expr(),
                                                      context]) \
                              for context in context_representations]

                nerIDs = [self.ner.get(tag) for tag in goldNers]

                loss = self.crf_module.neg_log_loss(tag_scores, nerIDs)
                # loss, _ = self.crf_module.viterbi_loss(tag_scores, nerIDs)

                nerErrs.append(loss)

                # observations = [dynet.concatenate([obs, dynet.inputVector([-1e10, -1e10])], d=0) for obs in
                #                tag_scores]
                # predicted_ner_indices, _ = self.crf_module.viterbi_decoding(observations)

                # Add ner embeddings
                for entry, ner in zip(sentence, nerIDs):
                    entry.vec = concatenate([entry.vec, dynet.dropout(self.nerlookup[ner], 0.33)])
                    entry.lstms = [entry.vec, entry.vec]

                # Relation losses
                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(sentence, reversed(sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(sentence, reversed(sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                concat_layer = [dynet.dropout(concatenate(entry.lstms), 0.33) for entry in sentence]

                head_context_representations = [dynet.tanh(dynet.affine_transform([self.head_layer_b.expr(),
                                                                                   self.head_layer_W.expr(),
                                                                                   context])) \
                                                for context in concat_layer]

                dep_context_representations = [dynet.tanh(dynet.affine_transform([self.dep_layer_b.expr(),
                                                                                  self.dep_layer_W.expr(),
                                                                                  context])) \
                                               for context in concat_layer]

                pairrels = id2arg2rel[sentenceId]
                for (head, dep) in pairrels:
                    # print (head, dep), pairrels[(head, dep)]
                    linear = self.rel_U.expr() * dep_context_representations[dep]
                    linear = dynet.reshape(linear, (self.hidden_units, len(self.rels)))
                    bilinear = dynet.transpose(head_context_representations[head]) * linear
                    biaffine = dynet.transpose(bilinear) + self.rel_W.expr() * concatenate(
                        [head_context_representations[head], dep_context_representations[dep]]) + self.rel_b.expr()
                    lerrs.append(self.pick_neg_log(softmax(biaffine), self.rels.get(pairrels[(head, dep)])))

                etotal += len(sentence)
                nwtotal += len(sentence)

                if iSentence % 1 == 0:
                    if len(lerrs) > 0 or len(nerErrs) > 0:
                        # if len(nerErrs) > 0:
                        eerrs = esum(nerErrs + lerrs)
                        eerrs.scalar_value()
                        eloss += eerrs.scalar_value()
                        mloss += eloss
                        eerrs.backward()
                        self.trainer.update()
                        # errs = []
                        lerrs = []
                        nerErrs = []

                    renew_cg()

        print "Loss: %.4f" % (mloss / nwtotal)

