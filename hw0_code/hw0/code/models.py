# CS505: NLP - Spring 2026
import torch
from collections import Counter, defaultdict
import random
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re
import numpy as np


class BoWFeaturizer:
    """
    This is a bag-of-words featurizer. It uses `build_vocab` to load a list of Examples
    and uses the top `max_vocab_size` words by frequency as its vocabulary.
    For a given Example, it counts the number of instances of each word in
    `self.vocab` and returns this vector of counts.
    """
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.vocab = {} # mapping word -> index
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, data):
        counts = Counter()
        # TODO: count the number instances of each token (here, just words and
        # punctuation) in `data`. Filter the vocab down to the `self.max_vocab_size` most
        # frequent tokens, and put these in a variable called `most_common`. HINT:
        # you can use the `word_tokenize` function that's been
        # imported above to tokenize the string.
        # STUDENT START ---------------------------------
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            counts.update(tokens)
        most_common = counts.most_common(self.max_vocab_size)
        # STUDENT END ------------------------------------

        
        # you might need to remove the `count` variable here, depending on how you
        # implemented the above.
        self.vocab = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary built with {self.vocab_size} words.")

    def get_feature_vector(self, text):
        pass
        # TODO: Return a bag-of-words feature vector. Each index in
        # the vocabulary should have a corresponding index in this vector.
        # A token's vector index should contain the frequency of that token
        # in `text`.
        # This shold return a torch tensor of size (vocab_size,).
        # STUDENT START -------------------------
        vec = torch.zeros(self.vocab_size)
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        return vec
        # STUDENT END ---------------------------


class BigramFeaturizer(BoWFeaturizer):
    def build_vocab(self, data):
        counts = Counter()
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            # TODO: generate bigrams
            # STUDENT START ----------------------------
            # 生成 bigrams: (word1, word2)
            bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
            counts.update(bigrams)
            # STUDENT END ------------------------------
        
        # TODO: build your vocabulary of the `self.max_vocab_size` most frequent bigrams.
        # STUDENT START -------------------------------------------
        most_common = counts.most_common(self.max_vocab_size)
        self.vocab = {bg: idx for idx, (bg, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: bg for idx, (bg, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)
        # STUDENT END ---------------------------------------------

    def get_feature_vector(self, text):
        tokens = word_tokenize(text.lower())
        vec = torch.zeros(self.vocab_size)
        
        # TODO: use the list of tokens to generate bigram features.
        # Return the bigram feature vector.
        # STUDENT START --------------------------------------
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i + 1]}"
            if bg in self.vocab:
                vec[self.vocab[bg]] += 1
        return vec
        # STUDENT END -----------------------------------------


class BlackBoxClassifier(torch.nn.Module):
    """
    This is a logistic regression classifier using PyTorch's built-in modules.
    Only used in Task 1. You will implement something like this from scratch
    in the LogisticRegressionClassifier class.
    """
    def __init__(self, input_dim, num_classes):
        super(BlackBoxClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.linear(x)


class LogisticRegressionClassifier:
    def __init__(self, input_dim, num_classes):
        # Initialize weights and bias
        # Weights: (input_dim, num_classes), Bias: (num_classes)
        self.weights = torch.randn(input_dim, num_classes, requires_grad=False) * 0.01
        self.bias = torch.zeros(num_classes, requires_grad=False)

    def forward(self, x):
        # TODO: implement the logistic regression as z = W^T * x + b. Return z.
        # Hint: this should only require one line of code!
        # STUDENT START ---------------------------------
        # z = xW + b (x 是 1xd, W 是 dxC)
        return torch.matmul(x, self.weights) + self.bias
        # STUDENT END -----------------------------------

    def softmax(self, logits):
        pass
        # TODO: implement softmax. You may *not* use torch.nn.softmax or any
        # similar function. You may use torch.exp if you wish.
        # STUDENT START --------------------------------
        # 减去 max 以保证数值稳定性
        exps = torch.exp(logits - torch.max(logits))
        return exps / torch.sum(exps)
        # STUDENT END ----------------------------------

    def predict(self, x):
        logits = self.forward(x)
        probs = self.softmax(logits)
        return torch.argmax(probs).item()


def train_logistic_regression(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5,
                              method="bow"):
    input_dim = featurizer.vocab_size
    if method == "lr":
        dummy_text = train_data[0].text
        dummy_vector = featurizer.get_feature_vector(dummy_text)
        input_dim = dummy_vector.shape[0]
        model = LogisticRegressionClassifier(input_dim, 4)
    elif method == "bow":
        model = BlackBoxClassifier(input_dim, num_classes)
    
    print("Training logistic regression...")
    
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text) # (vocab_size,)
            y_true = ex.label
            
            # 1. Call the forward function and compute the probability
            # of each class according to the model.
            logits = model.forward(x)
            probs = model.softmax(logits)
            
            # TODO: 2. Compute the negative log likelihood loss.
            # STUDENT START ----------------------------
            # 2. 计算负对数似然损失 (NLL Loss)
            loss = -torch.log(probs[y_true] + 1e-10)
            total_loss += loss.item()
            # STUDENT END ------------------------------
            
            # TODO: 3. Compute the gradient for the weights, and the gradient for
            # for the bias. You may not use .backward().
            # STUDENT START ----------------------------
            # 3. 计算梯度 (基于交叉熵求导结果)
            # dL/dz = probs - y_onehot
            target_onehot = torch.zeros(num_classes)
            target_onehot[y_true] = 1.0
            grad_z = probs - target_onehot
            # grad_W = x^T * grad_z (outer product)
            grad_w = torch.outer(x, grad_z)
            grad_b = grad_z


            # STUDENT END ------------------------------
            
            # TODO: 4. Update the parameters by multiplying the gradients you
            # derived in the previous step by the learning rate, and then subtracting
            # them from the weights and biases. You will need at least 1 line to update the
            # weight matrix, and at least 1 line to update the bias.
            # STUDENT START ----------------------------
            # 4. 更新参数
            model.weights -= lr * grad_w
            model.bias -= lr * grad_b
            # STUDENT END ------------------------------
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
        
    return model


def train_torch_model(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5):
    """
    Pre-provided gradient descent function using PyTorch's optimizer and loss.
    """
    input_dim = featurizer.vocab_size
    model = BlackBoxClassifier(input_dim, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("Training Built-in PyTorch Model...")
    
    # This is an example of a training loop. Here, we're using only black-box
    # built-in PyTorch functions. You will implement the underlying functionality
    # of these functions as part of Task 2.
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        model.train() # Set model to training mode
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text)
            x_tensor = x.unsqueeze(0)
            y_tensor = torch.tensor([ex.label], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
    
    # TODO: You're given the weight matrix of your trained model, which is of
    # shape (C, V), where C is the number of classes (here, 4) and V is
    # the vocabulary size. For each class, you will get the top-5 weight indices,
    # and print out the tokens they correspond to. No need to return anything here;
    # just print out the top weights/tokens and put them in your written report.
    # STUDENT START ----------------------------------
    weights = model.linear.weight
    label_names = ["world", "sports", "business", "tech"]
    for i in range(num_classes):
        top_indices = torch.topk(weights[i], 5).indices
        top_tokens = [featurizer.inverse_vocab[idx.item()] for idx in top_indices]
        print(f"Top 5 tokens for {label_names[i]}: {top_tokens}")
    # STUDENT END ------------------------------------
        
    return model


class CustomFeaturizer(BoWFeaturizer):
    def __init__(self):
        super().__init__()
        self.vocab_size = 5000
        # 基础 5000 维 + 3 个自定义特征维度
        self.output_dim = self.vocab_size + 3

    # build_vocab 直接继承父类，不需要重写

    def get_feature_vector(self, text):
        # 1. 获取基础 BoW 向量
        base_vector = super().get_feature_vector(text)

        # 2. 计算自定义特征

        # --- Feature A: Scoreboard Pattern (针对 Sports) ---
        # 匹配 "3-2", "105-98" 这类比分格式
        score_count = len(re.findall(r'\b\d{1,3}-\d{1,3}\b', text))

        # --- Feature B: Money Density (针对 Business) ---
        # 统计金钱符号
        money_count = text.count('$') + text.count('€') + text.count('£')

        # --- Feature C: Average Word Length (针对 Business vs Sports) ---
        words = text.split()
        if len(words) > 0:
            avg_len = sum(len(w) for w in words) / len(words)
        else:
            avg_len = 0.0

        custom_features = torch.tensor([
            np.log1p(score_count),
            np.log1p(money_count),
            avg_len
        ], dtype=torch.float32)

        # 4. 拼接 (Concatenate)
        # 结果维度: 5003
        final_vector = torch.cat((base_vector, custom_features), dim=0)

        return final_vector