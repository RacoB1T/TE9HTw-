"""
Shared test fixtures for LOGO tests.

Provides a simple mock tokenizer for offline testing when HuggingFace
tokenizers are unavailable.
"""

import pytest
from transformers import AutoTokenizer


_MOCK_VOCAB: dict = {}
_MAX_ID = 0


def _add_token(token: str) -> int:
    global _MAX_ID
    if token not in _MOCK_VOCAB:
        _MOCK_VOCAB[token] = _MAX_ID
        _MAX_ID += 1
    return _MOCK_VOCAB[token]


# Register Llama-3 special tokens
_add_token("<|begin_of_text|>")  # BOS
_add_token("<|end_of_text|>")    # EOS
_add_token("<|eot_id|>")         # EOT
_add_token("<|start_header_id|>")
_add_token("<|end_header_id|>")
_add_token("system")
_add_token("user")
_add_token("assistant")

# Common words
for w in [
    "Below", "are", "some", "references", ".", "Read", "them", "carefully",
    "and", "answer", "the", "question", "using", "References", ":\n",
    "[Chunk", "1", "2", "3", "]\n", "\n\n", "\n", "Question", ":\n",
    "Reference", "chunk", "Paris", "Berlin", "Rome", "The", "capital",
    "of", "France", "is", "Germany", "Italy", "Hello", "hello",
]:
    _add_token(w)

# Letters a-z
for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ":
    if c not in _MOCK_VOCAB:
        _add_token(c)


class MockTokenizer:
    """Minimal tokenizer for offline testing.

    Implements the interface needed by LogoDatasetBuilder and PromptAdapter.
    Does NOT subclass PreTrainedTokenizerBase to avoid complexity.
    """

    def __init__(self):
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"
        self.pad_token = "<|end_of_text|>"
        self.bos_token_id = _MOCK_VOCAB["<|begin_of_text|>"]
        self.eos_token_id = _MOCK_VOCAB["<|end_of_text|>"]
        self.pad_token_id = _MOCK_VOCAB["<|end_of_text|>"]
        self.unk_token_id = _MOCK_VOCAB["<|end_of_text|>"]
        self.name_or_path = "mock-llama3-tokenizer"
        self.model_max_length = 100000
        self._ids_to_tokens = {v: k for k, v in _MOCK_VOCAB.items()}

    def get_vocab(self):
        return dict(_MOCK_VOCAB)

    @property
    def vocab_size(self) -> int:
        return len(_MOCK_VOCAB)

    def _tokenize(self, text: str):
        """Simple word-then-character tokenization."""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                candidate = text[i:i + length]
                if candidate in _MOCK_VOCAB:
                    tokens.append(candidate)
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return _MOCK_VOCAB.get(token, _MOCK_VOCAB.get("<|end_of_text|>", 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, "<|end_of_text|>")

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(t) for t in tokens]

    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = self._tokenize(text)
        ids = [self._convert_token_to_id(t) for t in tokens]
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        tokens = []
        for tid in ids:
            token = self._convert_id_to_token(tid)
            if skip_special_tokens and token.startswith("<|"):
                continue
            tokens.append(token)
        return "".join(tokens)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        """Simulate Llama-3 chat template output."""
        parts = []
        if self.bos_token:
            parts.append(self.bos_token)

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
            if content:
                parts.append(content)
                parts.append("<|eot_id|>")

        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        text = "".join(parts)
        if tokenize:
            ids = self.encode(text, add_special_tokens=False)
            from types import SimpleNamespace
            return SimpleNamespace(input_ids=ids)
        return text


def _get_real_tokenizer(path: str):
    """Attempt to load a real tokenizer; return None on failure."""
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        return None


@pytest.fixture(scope="session")
def mock_tokenizer():
    """Returns a MockTokenizer for offline testing."""
    return MockTokenizer()


@pytest.fixture(scope="session")
def tokenizer_path():
    """Try real tokenizer, fall back to mock."""
    real = _get_real_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")
    if real is not None:
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    return "mock"  # Signal to use MockTokenizer
