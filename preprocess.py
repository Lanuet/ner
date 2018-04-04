import os
import itertools
import utils
import numpy as np


class Counter:
    def __init__(self):
        self.max_sen_len = 0
        self.longest_sen = None
        self.max_word_len = 0
        self.word_vocab = set()
        self.char_vocab = set()
        self.pos_tags = set()
        self.chunk_tags = set()
        self.ner_tags = set()

    def update(self, sen):
        self.max_sen_len = max(self.max_sen_len, len(sen))
        if self.max_sen_len == len(sen):
            self.longest_sen = sen
        self.max_word_len = max(self.max_word_len, sen.max_word_len)
        self.word_vocab = self.word_vocab | sen.word_vocab
        self.char_vocab = self.char_vocab | sen.char_vocab
        self.pos_tags = self.pos_tags | sen.pos_tags
        self.chunk_tags = self.chunk_tags | sen.chunk_tags
        self.ner_tags = self.ner_tags | sen.ner_tags

    def longest_word(self):
        sort = sorted(self.word_vocab, key=lambda w: len(w))
        return sort[-1]

    def __str__(self):
        data = {
            "max_sen_len": self.max_sen_len,
            "max_word_len": self.max_word_len,
            "word_vocab": len(self.word_vocab),
            "char_vocab": len(self.char_vocab),
        }
        return str(data)


class Word:
    def __init__(self, data):
        self.word, self.pos, self.chunk, self.ner, *_ = data
        self.chars = [c for c in self.word]
        self.char_vocab = set(self.chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return self.word

    def encode(self, encoder):
        chars_idx = np.zeros(encoder.max_word_len, dtype='int32')
        for i, c in enumerate(self.chars):
            chars_idx[i] = encoder.char2idx[c]
        word_idx = encoder.word2idx[self.word] if self.word in encoder.word2idx else encoder.word2idx["UNKNOWN"]
        pos_idx = encoder.pos2idx[self.pos]
        chunk_idx = encoder.chunk2idx[self.chunk]
        ner_idx = encoder.ner2idx[self.ner]
        return chars_idx, word_idx, pos_idx, chunk_idx, ner_idx


class Sentence:
    def __init__(self, words):
        self.words = [Word(w) for w in words]
        self.max_word_len = max(len(w) for w in self.words)
        self.word_vocab = set(w.word for w in self.words)
        self.char_vocab = [w.char_vocab for w in self.words]
        self.char_vocab = itertools.chain(*self.char_vocab)
        self.char_vocab = set(self.char_vocab)
        self.pos_tags = set(w.pos for w in self.words)
        self.chunk_tags = set(w.chunk for w in self.words)
        self.ner_tags = set(w.ner for w in self.words)

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return " ".join(map(str, self.words))

    def encode(self, encoder):
        chars = np.zeros([encoder.max_sen_len, encoder.max_word_len], dtype='int32')
        words = np.zeros(encoder.max_sen_len, dtype='int32')
        poss = np.zeros(encoder.max_sen_len, dtype='int32')
        chunks = np.zeros(encoder.max_sen_len, dtype='int32')
        ners = np.zeros(encoder.max_sen_len, dtype='int32')
        for i, w in enumerate(self.words):
            chars[i, :], words[i], poss[i], chunks[i], ners[i] = w.encode(encoder)
        return chars, words, poss, chunks, ners


def read_file(path, counter):
    sentences = utils.parse(utils.read(path), ["\n\n", "\n", "\t"])
    data = []
    for sentence in sentences:
        sentence = Sentence(sentence)
        counter.update(sentence)
        data.append(sentence)
    print("read %d sentences" % len(data))
    return data


def read_word_embedding(replace=False):
    unknown_dir = "embedding/unknown.npy"
    vectors_dir = "embedding/vectors.npy"
    words_dir = "embedding/words.json"
    word_embeddings_dir = "embedding/word_embeddings.npy"
    word2idx_dir = "embedding/word2idx.json"

    print("read word embedding")
    if replace or not os.path.exists(word_embeddings_dir) or not os.path.exists(word2idx_dir):
        vectors = np.load(vectors_dir)
        unknown = np.load(unknown_dir)
        extension = utils.get_file_extension(words_dir)[1:]
        assert extension in ["json", "pl"]
        if extension == "json":
            words = utils.json_load(words_dir)
        else:
            words = utils.pkl_load(words_dir)
        word2idx = {
            "UNKNOWN": 1,
            **{w: i+2 for i, w in enumerate(words)}
        }
        vectors = [
            unknown,
            *list(vectors),
        ]
        np.save(word_embeddings_dir, vectors)
        utils.json_dump(word2idx, word2idx_dir)
    else:
        word2idx = utils.json_load(word2idx_dir)
    print("vocab: %d words" % (len(word2idx) - 1))
    return word2idx


def tags2idx(tags):
    return {t: i + 1 for i, t in enumerate(tags)}


def encode_sens(sens, encoder):
    return list(map(np.array, zip(*[s.encode(encoder) for s in sens])))


def main():
    train_dir = "origin_data/-Doi_song_train.muc"
    dev_dir = "origin_data/Doi_song_dev.muc"
    test_dir = "origin_data/Doi_song_test.muc"

    counter = Counter()

    print("read train file")
    sentences_train = read_file(train_dir, counter)

    print("read dev file")
    sentences_dev = read_file(dev_dir, counter)

    print("read test file")
    sentences_test = read_file(test_dir, counter)

    print(counter)
    print("longest sentence: %s" % str(counter.longest_sen))
    print("longest word: %s" % counter.longest_word())

    word2idx = read_word_embedding()

    char2idx = tags2idx(counter.char_vocab)
    pos2idx = tags2idx(counter.pos_tags)
    chunk2idx = tags2idx(counter.chunk_tags)
    ner2idx = tags2idx(counter.ner_tags)
    utils.json_dump(char2idx, "embedding/char2idx.json")
    utils.json_dump(pos2idx, "embedding/pos2idx.json")
    utils.json_dump(chunk2idx, "embedding/chunk2idx.json")
    utils.json_dump(ner2idx, "embedding/ner2idx.json")

    print("encoding data")
    encoder = {
        "max_sen_len": counter.max_sen_len,
        "max_word_len": counter.max_word_len,
        **utils.make_dict(word2idx, char2idx, pos2idx, chunk2idx, ner2idx)
    }
    encoder = utils.dict_to_object(encoder)
    chars_train, words_train, pos_train, chunk_train, ner_train = encode_sens(sentences_train, encoder)
    chars_dev, words_dev, pos_dev, chunk_dev, ner_dev = encode_sens(sentences_dev, encoder)
    chars_test, words_test, pos_test, chunk_test, ner_test = encode_sens(sentences_test, encoder)

    print("saving data")
    data = utils.make_dict(chars_train, words_train, pos_train, chunk_train, ner_train, chars_dev, words_dev, pos_dev, chunk_dev, ner_dev, chars_test, words_test, pos_test, chunk_test, ner_test)
    os.makedirs("data", exist_ok=True)
    for k, d in data.items():
        np.save("data/%s.npy" % k, d)


if __name__ == '__main__':
    main()