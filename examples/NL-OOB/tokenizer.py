import pickle
from typing import List, Dict


class AminoAcidTokenizer:
    def __init__(self):
        # Standard 20 amino acids + extended
        self.vocab = [
            '[PAD]', '[UNK]',
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
            'X', 'B', 'Z', 'J', 'O', 'U'  # Rare/ambiguous
        ]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id['[PAD]']
        self.unk_token_id = self.token_to_id['[UNK]']

    def encode(self, text: str, max_len: int = None) -> List[int]:
        ids = []
        for char in text.upper():
            ids.append(self.token_to_id.get(char, self.unk_token_id))

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids += [self.pad_token_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            if i == self.pad_token_id:
                continue
            tokens.append(self.id_to_token.get(i, '[UNK]'))
        return "".join(tokens)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'AminoAcidTokenizer':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_vocab_size(self) -> int:
        return len(self.vocab)
