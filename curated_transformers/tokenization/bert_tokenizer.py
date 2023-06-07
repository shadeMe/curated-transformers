from typing import Any, Dict, Iterable, List, Optional, Set, Type, TypeVar
from curated_tokenizers import WordPieceProcessor
import json
from pathlib import Path
import unicodedata

from .chunks import InputChunks, SpecialPieceChunk, TextChunk
from .hf_hub import FromHFHub, FromPretrainedHFTokenizer
from .tokenizer import PiecesWithIds, PreEncoder, PostEncoder, PreDecoder, PostDecoder
from .util import remove_pieces_from_sequence
from .wordpiece_tokenizer import WordPieceTokenizer, clean_up_decoded_string_like_hf


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="BertTokenizer")


class BertPreEncoder(PreEncoder):
    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
        lowercase: bool,
        strip_accents: bool,
    ):
        """Construct a BERT pre-encoder.

        :param bos_piece:
            The piece used to mark the beginning of a sequence.
        :param eos_piece:
            The piece used to mark the end of a sequence.
        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Strip accents from text.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.lowercase = lowercase
        self.strip_accents = strip_accents

    def split_token_on_punctuation(self, token: str) -> List[str]:
        """Split a token on punctuation characters. For instance,
        'AWO-Mitarbeiter' is split into ['AWO', '-', 'Mitarbeiter']"""
        tokens = []
        in_word = False
        while token:
            char = token[0]
            token = token[1:]
            if self.is_punctuation(char):
                tokens.append([char])
                in_word = False
            else:
                if in_word:
                    tokens[-1].append(char)
                else:
                    tokens.append([char])
                    in_word = True
        return ["".join(t) for t in tokens]

    def is_punctuation(self, char: str) -> bool:
        """Checks whether `char` is a punctuation character."""
        # ASCII punctuation from HF tranformers, since we need to split
        # in the same way.
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True

        return unicodedata.category(char).startswith("P")

    def strip_token_accents(self, token: str) -> str:
        # TODO move this to the normalization phase of to the tokenizer
        token = unicodedata.normalize("NFD", token)
        return "".join([char for char in token if unicodedata.category(char) != "Mn"])

    def __call__(self, input: Iterable[InputChunks]) -> List[InputChunks]:
        preprocessed = []

        for seq in input:
            processed_seq = InputChunks([SpecialPieceChunk(self.bos_piece)])
            for chunk in seq:
                if isinstance(chunk, TextChunk):
                    words = []
                    for word in chunk.text.split(" "):
                        if self.lowercase:
                            word = word.lower()
                        if self.strip_accents:
                            word = self.strip_token_accents(word)
                        word_with_punct = self.split_token_on_punctuation(word)
                        words.extend(word_with_punct)
                    processed_seq.append(TextChunk(" ".join(words)))
                else:
                    processed_seq.append(chunk)
            processed_seq.append(SpecialPieceChunk(self.eos_piece))
            preprocessed.append(processed_seq)

        return preprocessed


class BertPostEncoder(PostEncoder):
    def __init__(
        self,
        *,
        unk_piece: str,
        unk_id: int,
    ):
        """Construct a BERT post-encoder.

        unk_piece (int): The piece used to mark unknown tokens.
        unk_id (int): The piece id used to mark unknown tokens.
        """
        self.unk_piece = unk_piece
        self.unk_id = unk_id

    def __call__(self, pieces_with_ids: PiecesWithIds) -> PiecesWithIds:
        # Replace all unknown IDs and their corresponding pieces.
        for ids, pieces in zip(pieces_with_ids.ids, pieces_with_ids.pieces):
            for i in range(len(ids)):
                piece_id = ids[i]
                if piece_id == -1:
                    ids[i] = self.unk_id
                    pieces[i] = self.unk_piece
        return pieces_with_ids


class BertPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """Construct a BERT pre-decoder.

        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]


class BertPostDecoder(PostDecoder):
    def __init__(
        self,
    ):
        """Construct a BERT post-decoder."""
        pass

    def __call__(self, output: Iterable[str]) -> List[str]:
        # The transformations done in the pre-encoding stage are lossy/non-reversible,
        # we can only do a selected number of transformations to massage the output
        # to look similar to the input text.
        return [clean_up_decoded_string_like_hf(string.strip()) for string in output]


class BertTokenizer(WordPieceTokenizer, FromHFHub, FromPretrainedHFTokenizer):
    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        special_pieces: Optional[Dict[str, int]] = None,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
    ):
        """Construct a Bert tokenizer from a curated tokenizers WordPiece processor.

        :param vocab:
            The word piece vocabulary.
        :param special_pieces:
            Special pieces.
        :param bos_piece:
            The piece used to mark the beginning of a sequence.
        :param eos_piece:
            The piece used to mark the end of a sequence.
        :param unk_piece:
            The piece used to mark unknown tokens.
        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Strip accents from text.
        """
        super().__init__(vocab=vocab, special_pieces=special_pieces)

        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.unk_piece = unk_piece

        bos_id = _get_piece_id_or_fail(self.processor, bos_piece)
        eos_id = _get_piece_id_or_fail(self.processor, eos_piece)
        unk_id = _get_piece_id_or_fail(self.processor, unk_piece)

        self.pre_encoder = BertPreEncoder(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
        )
        self.post_encoder = BertPostEncoder(
            unk_piece=unk_piece,
            unk_id=unk_id,
        )
        self.pre_decoder = BertPreDecoder(bos_id=bos_id, eos_id=eos_id)
        self.post_decoder = BertPostDecoder()

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
    ) -> Self:
        """Construct a tokenizer from the vocabulary file.

        vocab_path (Path): Path to the vocabulary file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        unk_piece (str): The piece used to mark unknown tokens.
        lowercase (bool): Lowercase text.
        strip_accents (bool): Strip accents from text.
        """
        vocab: Dict[str, int] = {}
        with open(vocab_path, encoding="utf8") as f:
            for line in f:
                vocab[line.strip()] = len(vocab)

        return cls(
            vocab=vocab,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
        )

    @classmethod
    def _convert_hf_tokenizer_json(
        cls: Type[Self], *, hf_tokenizer: Dict[str, Any]
    ) -> Self:
        if hf_tokenizer["pre_tokenizer"]["type"] != "BertPreTokenizer":
            raise ValueError(
                "Attempted to load a non-BERT tokenizer as a BERT tokenizer"
            )

        model = hf_tokenizer["model"]
        unk_piece = model["unk_token"]

        vocab = model["vocab"]
        special_pieces = {
            added["content"]: added["id"] for added in hf_tokenizer["added_tokens"]
        }

        normalizer = hf_tokenizer["normalizer"]
        lowercase = normalizer["lowercase"]
        strip_accents = normalizer["lowercase"]

        # Huggingface BERT also strips accents when lowercasing is enabled
        # and accent stripping is not defined.
        strip_accents = strip_accents or (strip_accents is not False and lowercase)

        special_tokens = hf_tokenizer["post_processor"]["special_tokens"]
        bos_piece = special_tokens["[CLS]"]["id"]
        eos_piece = special_tokens["[SEP]"]["id"]

        return cls(
            vocab=vocab,
            special_pieces=special_pieces,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
        )

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        return cls._convert_hf_tokenizer_json(hf_tokenizer=deserialized)

    def _special_tokens(self) -> Set[str]:
        special_tokens = {self.bos_piece, self.eos_piece, self.unk_piece}
        special_tokens.update(self.special_piece_to_id.keys())
        return special_tokens


def _get_piece_id_or_fail(processor: WordPieceProcessor, piece: str):
    try:
        return processor.get_initial(piece)
    except KeyError:
        raise ValueError(
            f"BERT piece encoder vocabulary doesn't contain '{piece}' piece"
        )
