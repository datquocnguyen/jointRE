# coding=utf-8
from collections import Counter
import os, re, codecs, string
import gzip
import pickle
from  BIOF1Validation import compute_Rel_f1, compute_NER_f1, compute_NER_f1_macro, compute_Rel_f1_macro
import numpy as np


def readCoNLL2004_prepared_corpus():
    # read pickled file
    data_in = open("corpus_prepared.pickled", 'rb')
    train_id2sent = pickle.load(data_in)
    train_id2pos = pickle.load(data_in)
    train_id2ner = pickle.load(data_in)
    train_id2nerBILOU = pickle.load(data_in)
    train_id2arg2rel = pickle.load(data_in)

    test_id2sent = pickle.load(data_in)
    test_id2pos = pickle.load(data_in)
    test_id2ner = pickle.load(data_in)
    test_id2nerBILOU = pickle.load(data_in)
    test_id2arg2rel = pickle.load(data_in)
    data_in.close()

    #train_id2nerBILOU = convertIOBEStoBIO(train_id2nerBILOU)
    #test_id2nerBILOU = convertIOBEStoBIO(test_id2nerBILOU)

    return train_id2sent, train_id2pos, train_id2ner, train_id2nerBILOU, train_id2arg2rel, test_id2sent, test_id2pos, test_id2ner , test_id2nerBILOU, test_id2arg2rel


def convertIOBEStoBIO(test_id2nerBILOU):
    """ Convert inplace IOBES encoding to BIO encoding """
    for index in test_id2nerBILOU:
        sentencetags = test_id2nerBILOU[index]
        sentencetags = re.sub(r"U-", 'B-', sentencetags)
        sentencetags = re.sub(r"S-", 'B-', sentencetags)
        sentencetags = re.sub(r"E-", 'I-', sentencetags)
        sentencetags = re.sub(r"L-", 'I-', sentencetags)
        test_id2nerBILOU[index] = sentencetags
    return test_id2nerBILOU

def convert2BIO(tag):
    tag = re.sub(r"U-", 'B-', tag)
    tag = re.sub(r"S-", 'B-', tag)
    tag = re.sub(r"E-", 'I-', tag)
    tag = re.sub(r"L-", 'I-', tag)
    return tag

def vocabNER(train_id2sent, train_id2pos, train_id2nerBILOU):
    wordsCount = Counter()
    nerCount = Counter()
    posCount = Counter()
    # Character vocabulary
    c2i = {}
    c2i["_UNK"] = 0  # unk char
    c2i["<w>"] = 1  # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    c2i["EMAIL"] = 4
    c2i["URL"] = 5
    for ind in train_id2sent.keys():
        words = train_id2sent[ind].strip().split()
        wordsCount.update([normalize(word) for word in words])
        poses = train_id2pos[ind].strip().split()
        posCount.update([pos for pos in poses])
        for word in words:
            for char in word:
                if char not in c2i:
                    c2i[char] = len(c2i)

        nertags = train_id2nerBILOU[ind].strip().split()
        nerCount.update([tag for tag in nertags])

    return wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, c2i, nerCount.keys(), posCount


def getRelVocab(train_id2arg2rel, train_id2nerBILOU):
    id2arg2rel = {}
    relVocab = {}
    classcounter = Counter()
    for index in train_id2arg2rel:
     #   print("***")
        nerids = []
        tags = train_id2nerBILOU[index].strip().split()
        for ind, tag in enumerate(tags):
            #print(ind, tag)
            if str(tag).startswith("L-") or str(tag).startswith("U-"):
                if not str(tag).endswith("-Other"):
                    nerids.append(ind)

    #    print nerids, tags

        id2arg2rel[index] = {}
        for ind1 in nerids:
            for ind2 in nerids:
                if ind1 != ind2:
                    if tags[ind1] in ["L-Peop", "U-Peop"] and tags[ind2] in ["L-Peop", "U-Peop", "L-Org", "U-Org", "L-Loc", "U-Loc"]:
                        id2arg2rel[index][(ind1, ind2)] = "NEG"
                    if tags[ind1] in ["L-Loc", "U-Loc", "L-Org", "U-Org"] and tags[ind2] in ["L-Loc", "U-Loc"]:
                        id2arg2rel[index][(ind1, ind2)] = "NEG"

        #print id2arg2rel[index]
    #    print train_id2arg2rel[index]

        for key in id2arg2rel[index]:
            if key in train_id2arg2rel[index]:
                id2arg2rel[index][key] = train_id2arg2rel[index][key]

        classcounter.update([id2arg2rel[index][key] for key in id2arg2rel[index]])
            #(ind1, ind2) = key
            #if (ind2, ind1) in train_id2arg2rel[index]:
            #    id2arg2rel[index][key] = train_id2arg2rel[index][(ind2, ind1)] + "-1"

    #    print id2arg2rel[index]

    #print relVocab
        for key in id2arg2rel[index]:
            relVocab[id2arg2rel[index][key]] = True

    majority = max(classcounter.values())
    classweights = {cls: float(majority) / count for cls, count in classcounter.items()}

    return id2arg2rel, relVocab.keys(), classweights

class Word:
    def __init__(self, form, pos):
        self.form = form
        self.norm = normalize(form)
        self.idChars = []
        self.pred_ner = None
        self.pos = pos

    def __str__(self):
        return self.form + "/" +  self.norm

def readData(train_id2sent, train_id2pos, c2i):
    data = {}
    for index in train_id2sent:
        words = train_id2sent[index].strip().split()
        poses = train_id2pos[index].strip().split()
        tokens = []
        for word, pos in zip(words, poses):

            entry = Word(word, pos)

            if entry.norm == 'NUM':
                entry.idChars = [1, 3, 2]
            elif entry.norm == 'EMAIL':
                entry.idChars = [1, 4, 2]
            elif entry.norm == 'URL':
                entry.idChars = [1, 5, 2]
            else:
                chars_of_word = [1]
                for char in word:
                    if char in c2i:
                        chars_of_word.append(c2i[char])
                    else:
                        chars_of_word.append(0)
                chars_of_word.append(2)
                entry.idChars = chars_of_word

            tokens.append(entry)

        data[index] = tokens
        #print [data[index][i].pos for i in range(len(data[index]))]
    return data

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    if numberRegex.match(word):
        return 'NUM'
    elif word == "-LRB-":
        return '('
    elif word == "-RRB-":
        return ')'
    elif word == "COMMA":
        return ','
    else:
        w = word.lower()
        w = re.sub(r".+@.+", "EMAIL", w)
        w = re.sub(r"(https?://|www\.).*", "URL", w)
        w = re.sub(r"``", '"', w)
        w = re.sub(r"''", '"', w)
        return w

#try:
#    import lzma
#except ImportError:
#    from backports import lzma

def load_embeddings_file(file_name, lower=False):
        """
        Load embeddings file. Uncomment comments above and below if file format is .xz
        """
        if not os.path.isfile(file_name):
            print(file_name, "does not exist")
            return {}, 0

        emb={}
        print("Loading pre-trained word embeddings: {}".format(file_name))

        open_func = codecs.open

        #if file_name.endswith('.xz'):
        #    open_func = lzma.open
        #else:
        #    open_func = codecs.open

        if file_name.endswith('.gz'):
            open_func = gzip.open

        with open_func(file_name, 'rb') as f:
            reader = codecs.getreader('utf-8')(f, errors='ignore')
            reader.readline()

            count = 0
            for line in reader:
                try:
                    fields = line.strip().split()
                    vec = [float(x) for x in fields[1:]]
                    word = fields[0]
                    if lower:
                        word = word.lower()
                    if word not in emb:
                        emb[word] = vec
                except ValueError:
                    #print("Error converting: {}".format(line))
                    pass

                count += 1
                if count >= 1500000:
                    break
        return emb, len(emb[word])

def orthonormal_initializer(output_size, input_size):
	"""
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
	"""
	print (output_size, input_size)
	I = np.eye(output_size)
	lr = .1
	eps = .05/(output_size + input_size)
	success = False
	tries = 0
	while not success and tries < 10:
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
		for i in xrange(100):
			QTQmI = Q.T.dot(Q) - I
			loss = np.sum(QTQmI**2 / 2)
			Q2 = Q**2
			Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
			if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
				tries += 1
				lr /= 2
				break
		success = True
	if success:
		print('Orthogonal pretrainer loss: %.2e' % loss)
	else:
		print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
	return np.transpose(Q.astype(np.float32))


if __name__ == '__main__':
    train_id2sent, train_id2pos, train_id2ner, train_id2nerBILOU, train_id2arg2rel, test_id2sent, test_id2pos, test_id2ner, test_id2nerBILOU, test_id2arg2rel = readCoNLL2004_prepared_corpus()
    id2arg2rel, relVocab, classcounter = getRelVocab(test_id2arg2rel, test_id2nerBILOU)
    words, w2i, c2i, vocabnertags, postags = vocabNER(train_id2sent, train_id2pos, train_id2nerBILOU)
    data = readData(test_id2sent, test_id2pos, c2i)


    #(nerpred, relpred) = pickle.load(open( "output.pred_bk", "rb" ))
    (predDev, relpred) = pickle.load(open("outputs/output.predtest_ep29", "rb"))
    for index in test_id2sent:
        print test_id2sent[index]
        print test_id2ner[index]
        print test_id2nerBILOU[index]
        print test_id2arg2rel[index]
        print id2arg2rel[index]
        print " ".join([entry.norm for entry in data[index]])
        print "---"
        print " ".join(predDev[index])
        print relpred[index]
        print "------"

    #pre_rel, rec_rel, f1_rel = compute_Rel_f1(relpred, test_id2arg2rel)
    #print "RC results on test set - pre: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(pre_rel, rec_rel, f1_rel)

    #print classcounter

    # print(majority)

    label_pred = []
    label_correct = []
    for sentenceID in predDev:
        label_pred.append(predDev[sentenceID])
        label_correct.append(test_id2nerBILOU[sentenceID].strip().split())

    assert len(label_pred) == len(label_correct)

    f1 = compute_NER_f1_macro(label_pred, label_correct, 'O', "IOBES")
    f1_b = compute_NER_f1_macro(label_pred, label_correct, 'B', "IOBES")

    if f1_b > f1:
        #logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
        f1 = f1_b

    print "NER F1 results on test set - pre: {:.4f}".format( f1)

    f1_rel = compute_Rel_f1_macro(relpred, test_id2arg2rel)
    print "RC F1 results on test set - pre: {:.4f}".format(f1_rel)

    writer = open("output1.txt", "w")

    for index in test_id2sent:
        words = test_id2sent[index].split()
        poses = test_id2pos[index].split()
        nertags = [convert2BIO(tag) for tag in test_id2nerBILOU[index].split()]
        nerpredtags = [convert2BIO(tag) for tag in predDev[index]]

        for word, pos, ner, predner in zip(words, poses, nertags, nerpredtags):
            writer.write(" ".join([word, pos, ner, predner]) + "\n")

        writer.write("\n")

    writer.close()

    #print vocabnertags
    print relVocab