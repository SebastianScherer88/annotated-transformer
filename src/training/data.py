import os
from os.path import exists
import torch
from torch.nn.functional import pad
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
from typing import Tuple, List
from torch.utils.data.distributed import DistributedSampler
from training import SpecialTokens

class Preprocessor(object):
    
    tokenizer_src: spacy.Tokenizer
    tokenizer_tgt: spacy.Tokenizer
    vocab_src: torch.utils.Vocabulary
    vocab_tgt: torch.utils.Vocabulary
    vocab_src_size: int
    vocab_tgt_size: int
    
    def __init__(self, language_src: str = "de", language_tgt: str="en", max_padding:int=128):
        self.language_src = language_src
        self.language_tgt = language_tgt
        self.tokenizer_src, self.tokenizer_tgt = self.load_tokenizers()
        self.vocab_src, self.vocab_tgt = self.load_vocabularies()
        self.vocab_src_size, self.vocab_tgt_size = len(self.vocab_src), len(self.vocab_tgt)
        self.max_padding = max_padding
    
    def load_tokenizers(self) -> Tuple[spacy.Tokenizer,spacy.Tokenizer]:
        """Returns spacy tokenizers for the source and target language."""

        tokenizers = []

        for language in (self.language_src,self.language_tgt):
            try:
                tokenizer = spacy.load(f"{language}_core_news_sm")
            except IOError:
                os.system(f"python -m spacy download {language}_core_news_sm")
                tokenizer = spacy.load(f"{language}_core_news_sm")
            tokenizers.append(tokenizer)

        return tokenizers

    def tokenize_src(self, text):
        return [tok.text for tok in self.tokenizer_src.tokenizer(text)]
    
    def tokenize_tgt(self, text):
        return [tok.text for tok in self.tokenizer_tgt.tokenizer(text)]

    @staticmethod
    def yield_tokens(data_iter, tokenizer, index):
        for from_to_tuple in data_iter:
            yield tokenizer(from_to_tuple[index])
        
    def build_vocabularies(self):

        print("Building German Vocabulary ...")
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))
        vocab_src = build_vocab_from_iterator(
            self.yield_tokens(train + val + test, self.tokenize_src, index=0),
            min_freq=2,
            specials=SpecialTokens.list(),
        )

        print("Building English Vocabulary ...")
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))
        vocab_tgt = build_vocab_from_iterator(
            self.yield_tokens(train + val + test, self.tokenize_tgt, index=1),
            min_freq=2,
            specials=SpecialTokens.list(),
        )

        vocab_src.set_default_index(vocab_src[SpecialTokens.unk.value])
        vocab_tgt.set_default_index(vocab_tgt[SpecialTokens.unk.value])

        return vocab_src, vocab_tgt


    def load_vocabularies(self):
        if not exists("vocab.pt"):
            vocab_src, vocab_tgt = self.build_vocabulary(self.tokenize_src, self.tokenize_tgt)
            torch.save((vocab_src, vocab_tgt), "vocab.pt")
        else:
            vocab_src, vocab_tgt = torch.load("vocab.pt")
        print("Finished.\nVocabulary sizes:")
        print(len(vocab_src))
        print(len(vocab_tgt))
        return vocab_src, vocab_tgt

    def collate_batch(
        self,
        batch: List[Tuple[str,str]],
        device,
        max_padding=128,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        bs_id = torch.tensor([self.vocab_src.get_stoi(SpecialTokens.start.value)], device=device)  # <s> token id
        eos_id = torch.tensor([self.vocab_src.get_stoi(SpecialTokens.end.value)], device=device)  # </s> token id
        src_list, tgt_list = [], []
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_src(self.tokenize_src(_src)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_tgt(self.tokenize_tgt(_tgt)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        max_padding - len(processed_src),
                    ),
                    value=self.vocab_src.stoi(SpecialTokens.blank.value),
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_padding - len(processed_tgt)),
                    value=self.vocab_tgt.stoi(SpecialTokens.blank.value),
                )
            )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return (src, tgt)

def create_dataloaders(
    device,
    preprocessor: Preprocessor,
    batch_size=12000,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def collate_fn(batch):
        return preprocessor.collate_batch(
            batch,
            device,
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader