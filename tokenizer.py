"""
Module de tokenization / détokenization.
Gère la conversion texte ↔ tokens avec un vocabulaire BPE.
"""

import re
import json
import pickle
from typing import List, Tuple, Dict, Optional
from collections import Counter
import hashlib


class Tokenizer:
    """
    Tokenizer BPE (Byte-Pair Encoding) avancé.
    Supportclass Tokenizer:
    """
    Tokenizer BPE (Byte-Pair Encoding) avancé.
    Supporte la tokenization, détokenization et la gestion de vocabulaire.
    """

    def __init__(self, vocab_path: Optional[str] = None, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()
        self.vocab_size = self.model_config.vocab_size
        self.context_window = self.model_config.context_window
        self.max_new_tokens = self.model_config.max_new_tokens

        self.encoder: Dict[int, str] = {}
        self.decoder: Dict[str, int] = {}
        self.pat_tokenizer = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.bpe_ranks: Dict[Tuple[str, ...], int] = {}
        self.cache: Dict[str, List[int]] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4,
            "<bos>": 5,
            "<eos>": 6,
        }
        self.added_tokens: Dict[str, int] = {}
        self._initialized = False

        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Construit un vocabulaire par défaut pour l'initialisation."""
        for i in range(256):
            char = chr(i)
            self.encoder[i] = char
            self.decoder[char] = i

        idx = 256
        common_patterns = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        ]
        for pattern in common_patterns:
            if idx < self.vocab_size:
                self.encoder[idx] = pattern
                self.decoder[pattern] = idx
                idx += 1

        self._initialized = True
        self._build_bpe_ranks()

    def _build_bpe_ranks(self):
        """Construit les classements BPE pour la compression."""
        pairs = Counter()
        for token_id, token_str in self.encoder.items():
            if len(token_str) > 1:
                chars = list(token_str)
                for i in range(len(chars) - 1):
                    pairs[(chars[i], chars[i + 1])] += 1

        sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])
        self.bpe_ranks = {pair: rank for rank, (pair, _) in enumerate(sorted_pairs)}

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Retourne les paires de caractères adjacentes."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token: str) -> List[int]:
        """Applique l'algorithme BPE à un token."""
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        if len(word) <= 1:
            return [self.encoder.get(self._utf8_byte_encode(token), 3)]

        pairs = self._get_pairs(word)
        if not pairs:
            return [self.encoder.get(self._utf8_byte_encode(token), 3)]

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        word_str = ' '.join(word)
        token_ids = []
        for w in word:
            encoded = self._utf8_byte_encode(w)
            token_id = self.encoder.get(encoded, 3)
            token_ids.append(token_id)

        self.cache[token] = token_ids
        return token_ids

    def _utf8_byte_encode(self, text: str) -> str:
        """Encode le texte en bytes UTF-8 représentés comme chaîne."""
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convertit du texte en liste d'IDs de tokens.

        Args:
            text: Texte à tokenizer.
            add_special_tokens: Ajouter les tokens spéciaux de début/fin.

        Returns:
            Liste des IDs de tokens.
        """
        if not text:
            return [self.special_tokens["<bos>"]] if add_special_tokens else []

        all_tokens = []
        for part in self.pat_tokenizer.findall(text):
            part_tokens = self._bpe(part)
            all_tokens.extend(part_tokens)

        if add_special_tokens:
            bos_id = self.special_tokens.get("<bos>", 1)
            eos_id = self.special_tokens.get("</s>", 2)
            all_tokens = [bos_id] + all_tokens + [eos_id]

        if len(all_tokens) > self.context_window:
            all_tokens = all_tokens[:self.context_window]

        return all_tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Convertit une liste d'IDs de tokens en texte.

        Args:
            token_ids: Liste des IDs de tokens.

        Returns:
            Texte décodé.
        """
        text_parts = []
        for token_id in token_ids:
            if token_id in self.special_tokens.values():
                continue
            token_str = self.encoder.get(token_id, "")
            if token_str:
                text_parts.append(token_str)

        text = ''.join(text_parts)
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except:
            pass
        return text

    def encode_with_offsets(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Tokenize avec les positions de début et fin dans le texte original.

        Returns:
            Liste de tuples (token_id, start, end).
        """
        tokens = []
        current_pos = 0
        for match in self.pat_tokenizer.finditer(text):
            token_text = match.group()
            start = match.start()
            end = match.end()
            token_ids = self._bpe(token_text)
            for tid in token_ids:
                tokens.append((tid, start, end))
            current_pos = end
        return tokens

    def add_tokens(self, new_tokens: List[str]) -> int:
        """
        Ajoute de nouveaux tokens au vocabulaire.

        Returns:
            Nombre de tokens ajoutés.
        """
        added = 0
        for token in new_tokens:
            if token not in self.decoder and len(self.encoder) < self.vocab_size:
                new_id = len(self.encoder)
                self.encoder[new_id] = token
                self.decoder[token] = new_id
                self.added_tokens[token] = new_id
                added += 1
        self._build_bpe_ranks()
        self.cache.clear()
        return added

    def save_vocab(self, path: str):
        """Sauvegarde le vocabulaire dans un fichier."""
        vocab_data = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "bpe_ranks": {str(k): v for k, v in self.bpe_ranks.items()},
            "special_tokens": self.special_tokens,
            "added_tokens": self.added_tokens,
            "cache": self.cache,
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)

    def load_vocab(self, path: str):
        """Charge un vocabulaire depuis un fichier."""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        self.encoder = vocab_data["encoder"]
        self.decoder = vocab_data["decoder"]
        self.bpe_ranks = {tuple(eval(k)): v for k, v in vocab_data["bpe_ranks"].items()}
        self.special_tokens = vocab_data["special_tokens"]
        self.added_tokens = vocab_data["added_tokens"]
        self.cache = vocab_data["cache"]
        self._initialized = True

    def __len__(self) -> int:
        return len(self.encoder)

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={len(self)}, context_window={self.context_window
