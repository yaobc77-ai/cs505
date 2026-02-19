import argparse
import math
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tokenizer import BPETokenizer, WordTokenizer


def train_neural_model(model, train_data, vocab_size, epochs=2, batch_size=32, lr=0.001, device='cpu'):
    # Create dataset (Batch, Seq_Len)
    seq_len = 30
    x_list, y_list = [], []
    for i in range(0, len(train_data) - seq_len, seq_len):
        chunk = train_data[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            continue
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])  # Next token prediction

    X = torch.tensor(x_list, dtype=torch.long).to(device)
    Y = torch.tensor(y_list, dtype=torch.long).to(device)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()  # Since models return log_softmax
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for bx, by in loader:
            optimizer.zero_grad()
            log_probs, _ = model(bx)  # Output: (Batch, Seq, Vocab)

            # Flatten for loss
            loss = criterion(log_probs.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f} | Time: {time.time() - start_time:.2f}s")


class NGramLM:
    def __init__(self, n, k=1.0):
        self.n = n
        self.k = k
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.vocab = set()

    def get_ngrams(self, tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def train(self, tokens):
        # 遍历所有的 n-gram
        for ngram in self.get_ngrams(tokens, self.n):
            context = tuple(ngram[:-1])  # 前 n-1 个 token 是上下文
            token = ngram[-1]  # 最后一个 token 是要预测的词

            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

            # 将见过的 token 加入词汇表
            for t in ngram:
                self.vocab.add(t)

    def get_prob(self, context, token):
        ngram = tuple(list(context) + [token])
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        V = len(self.vocab)

        # Laplace 平滑公式
        prob = (count_ngram + self.k) / (count_context + self.k * V)
        return prob

    def perplexity(self, test_tokens):
        log_prob_sum = 0.0
        ngrams = self.get_ngrams(test_tokens, self.n)
        N = len(ngrams)

        # 计算所有 n-gram 的对数概率之和
        for ngram in ngrams:
            context = tuple(ngram[:-1])
            token = ngram[-1]
            prob = self.get_prob(context, token)
            log_prob_sum += math.log(prob)

        # Perplexity = exp(- (1/N) * sum(log P))
        ppl = math.exp(-log_prob_sum / N)
        return ppl


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=100):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # RNN Parameters
        self.W_ih = nn.Parameter(torch.Tensor(embed_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def rnn_cell(self, x, h_prev):
        h_t = torch.tanh(x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh)
        return h_t

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []

        # 1. 词向量嵌入
        embedded = self.embedding(x)

        # 2. 遍历序列长度，计算隐藏状态
        h_t = hidden
        for i in range(seq_len):
            x_t = embedded[:, i, :]
            h_t = self.rnn_cell(x_t, h_t)
            outputs.append(h_t.unsqueeze(1))

            # 3. 拼接隐藏状态并计算 Logits
        outputs = torch.cat(outputs, dim=1)
        logits = self.fc(outputs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs, h_t

    def get_perplexity(self, data_loader, device='cpu'):
        self.eval()
        total_nll = 0
        total_tokens = 0
        criterion = nn.NLLLoss(reduction='sum')

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                log_probs, _ = self(x)
                loss = criterion(log_probs.view(-1, self.vocab_size), y.view(-1))
                total_nll += loss.item()
                total_tokens += y.numel()

        return math.exp(total_nll / total_tokens)


# Extra credit!
class LSTMLM(RNNLM):
    def __init__(self, vocab_size, embed_size=100, hidden_size=100):
        super(LSTMLM, self).__init__(vocab_size, embed_size, hidden_size)

        # LSTM Parameters: 4 gates
        self.W_ih_lstm = nn.Parameter(torch.Tensor(embed_size, 4 * hidden_size))
        self.W_hh_lstm = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b_ih_lstm = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh_lstm = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_weights()

    def lstm_cell(self, x, h_prev, c_prev):
        gates = x @ self.W_ih_lstm + self.b_ih_lstm + h_prev @ self.W_hh_lstm + self.b_hh_lstm
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def forward(self, x, states=None):
        batch_size, seq_len = x.size()
        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = states

        embedded = self.embedding(x)
        outputs = []

        for i in range(seq_len):
            x_t = embedded[:, i, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        logits = self.fc(outputs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs, (h_t, c_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["NGRAM", "RNN", "LSTM"])
    args = parser.parse_args()

    print(f"Running {args.model} Language Model...")

    # Load train / dev / test data
    print("Loading data...")
    with open("../train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    with open("../dev.txt", "r", encoding="utf-8") as f:
        dev_text = f.read()
    with open("../test.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    # Tokenization (train ONLY)
    start_train = time.time()

    print("Tokenizing data...")
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(train_text)

    # Q2 测试手动分词
    test_words = ["Boston", "University", "unhappiness", "running"]
    print("\n--- Q2: Manual Tokenization Examples ---")
    for word in test_words:
        ids = tokenizer.tokenize(word)
        # 获取 token 列表，处理未知 token
        tokens_list = [tokenizer.inverse_vocab.get(i, "<unk>") for i in ids]
        print(f"Word: '{word}' -> Tokens: {tokens_list}")
    print("----------------------------------------\n")

    train_data = tokenizer.tokenize(train_text)
    dev_data = tokenizer.tokenize(dev_text)
    test_data = tokenizer.tokenize(test_text)

    vocab_size = len(tokenizer.vocab)
    print(f"Vocab Size: {vocab_size}")

    # N-gram Models
    if args.model == "NGRAM":
        lm = NGramLM(n=3, k=1.0)
        lm.train(train_data)

        train_time = time.time() - start_train

        start_eval = time.time()
        dev_ppl = lm.perplexity(dev_data)
        test_ppl = lm.perplexity(test_data)

        print(f"Training Time: {train_time:.2f}s")
        print(f"Dev Perplexity:  {dev_ppl:.4f}")
        print(f"Test Perplexity: {test_ppl:.4f}")

    # Neural Models
    elif args.model in ["RNN", "LSTM"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"


        def make_loader(data):
            seq_len = 30
            x, y = [], []
            for i in range(0, len(data) - seq_len, seq_len):
                chunk = data[i:i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    continue
                x.append(chunk[:-1])
                y.append(chunk[1:])
            return DataLoader(
                TensorDataset(torch.tensor(x), torch.tensor(y)),
                batch_size=32,
                shuffle=False
            )


        dev_loader = make_loader(dev_data)
        test_loader = make_loader(test_data)

        if args.model == "RNN":
            model = RNNLM(vocab_size).to(device)
            train_neural_model(model, train_data, vocab_size, device=device)

        elif args.model == "LSTM":
            model = LSTMLM(vocab_size).to(device)
            train_neural_model(model, train_data, vocab_size, device=device)

        train_time = time.time() - start_train

        start_eval = time.time()
        dev_ppl = model.get_perplexity(dev_loader, device=device)
        test_ppl = model.get_perplexity(test_loader, device=device)

        print(f"Training Time: {train_time:.2f}s")
        print(f"Dev Perplexity:  {dev_ppl:.4f}")
        print(f"Test Perplexity: {test_ppl:.4f}")