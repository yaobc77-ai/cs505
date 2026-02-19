# CS505: NLP - Spring 2026

from nltk.tokenize import word_tokenize
import collections
import re


class WordTokenizer:
    """
    A simple baseline tokenizer that splits on whitespace and handles unknown tokens.
    Use this if you get stuck on the BPE implementation and decide to implement
    the language models first.
    """

    def __init__(self, vocab_size=None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self.vocab_size = vocab_size

    def train(self, text):
        # 1. Count word frequencies
        word_counts = collections.Counter(word_tokenize(text))

        # 2. Determine vocabulary size
        # If vocab_size is not set, we use all unique words + specials
        num_words_to_keep = len(word_counts)
        if self.vocab_size is not None:
            num_words_to_keep = self.vocab_size - len(self.special_tokens)

        # 3. Get most common words
        most_common = word_counts.most_common(num_words_to_keep)

        # 4. Initialize vocab with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        # 5. Add common words to vocab
        current_id = len(self.vocab)
        for word, _ in most_common:
            if word not in self.vocab:
                self.vocab[word] = current_id
                self.inverse_vocab[current_id] = word
                current_id += 1

        print(f"WordTokenizer training complete. Vocab size: {len(self.vocab)}")

    def tokenize(self, text):
        """
        Converts text to IDs. Words not in vocab are mapped to <unk>.
        """
        unk_id = self.vocab["<unk>"]
        tokens = []
        for word in word_tokenize(text):
            tokens.append(self.vocab.get(word, unk_id))
        return tokens


class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}  # Map token -> id
        self.inverse_vocab = {}  # Map id -> token

        # merges learned in order: list of ((a,b), "ab")
        self.merges = []
        # rank lookup: (a,b) -> rank (lower = earlier = higher priority)
        self.bpe_ranks = {}

        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

    def get_stats(self, vocab_counts):
        pairs = collections.defaultdict(int)
        for word, freq in vocab_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        """
        Merge the most frequent pair in the vocabulary.
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text):
        print("Training BPE Tokenizer...")

        # 1. 基础分词与字符拆分
        word_freqs = collections.Counter(word_tokenize(text))
        vocab_counts = {" ".join(list(word) + ["▁"]): count for word, count in word_freqs.items()}

        # 提取基础字符
        alphabet = set()
        for word in word_tokenize(text):
            for char in word:
                alphabet.add(char)
        alphabet.add("▁")

        # 2. 迭代合并最频繁的 token 对
        current_vocab_size = len(self.special_tokens) + len(alphabet)

        while current_vocab_size < self.vocab_size:
            pairs = self.get_stats(vocab_counts)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            vocab_counts = self.merge_vocab(best_pair, vocab_counts)
            current_vocab_size += 1

            # 进度打印
            if len(self.merges) % 100 == 0:
                print(f"Merges: {len(self.merges)} | Vocab Size: {current_vocab_size}")

        # 3. 更新词汇表
        self.vocab = {}
        self.inverse_vocab = {}

        # 添加特殊 tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        # 添加基础字符
        sorted_alphabet = sorted(list(alphabet))
        for char in sorted_alphabet:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char

        # 添加合并生成的词
        for (p0, p1) in self.merges:
            new_token = p0 + p1
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

        # 更新 rank 字典，供 _apply_bpe 使用
        self.bpe_ranks = dict(zip(self.merges, range(len(self.merges))))

        print(f"Training complete. Final Vocab size: {len(self.vocab)}")

    def _apply_bpe(self, symbols):
        """
        Apply BPE merges to a list of symbols, in correct priority order.
        symbols: List[str]
        returns: List[str]
        """
        # keep merging until no mergeable pairs exist
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # pick best ranked pair among those present
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break

            _, best_pair = min(ranked)  # smaller rank = earlier learned

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols

    def tokenize(self, text):
        tokens = []

        for word in word_tokenize(text):
            # same pre-tokenization scheme as training
            symbols = list(word) + ["▁"]

            # apply merges in order
            symbols = self._apply_bpe(symbols)

            tokens.extend(symbols)

        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        return ids