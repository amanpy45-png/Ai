class Tokenizer:
    def __init__(self, text):
        self._build_vocab(text)
        self._build_stoi()
        self._build_itos()

    def _build_vocab(self, text):
        self.vocab = sorted(list(set(text)))

    def _build_stoi(self):
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    def _build_itos(self):
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.vocab)