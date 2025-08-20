import regex as re
from collections import Counter
from collections import defaultdict
import unicodedata
from heapq import nlargest

naija_eng_pattern = re.compile(r"""
(
    \s+
  | \p{L}[\p{L}\p{M}'’\-]*(?!\p{M})



  | \p{N}+
  | [\p{P}\p{S}]+
  | .
)
""", re.UNICODE | re.VERBOSE)
special_tokens = {
    "<|startoftext|>": 70000,
    "<|endoftext|>": 70001,
}


class Batched_size_BpeTokenizer:
    def __init__(self,vocab_size,stop_list_size = 100):
        assert vocab_size >= 256
        self.vocab_size = vocab_size
        self.stop_words = None
        self.stop_list_size = stop_list_size
        self.split_pattern = naija_eng_pattern
        self.compiled_pattern = re.compile(self.split_pattern)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens  #for encoding 
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()} # for decoding 
  

    def texts_to_ids_freq_table(self, text_list,counts):
        text_list = text_list(re.findall(self.compiled_pattern, str(text_list)))

        """
        This function will transform our text into a list of tuples, each tuple
        containing a list of all the byte-objects of each text and its frequency 
        If memory/speed bottleneck occurs frequency_cutoffs will be used   
        """

        counts = Counter() if counts is None else counts # !!!! use of regex
        counts.update(text_list)
        
        """deriving stop words (frequent ariticles)"""
        if self.stop_list_size:
            self.stop_words = {counts.most_common(self.stop_list_size*2)} 
            # all that is flet is to change the value to the index of our vocab

        return [([*key.encode('utf-8')], val) for key,val in counts.items() if key not in self.stop_words]
    
    def get_stats (self, ids_freq_table):
        pair_counts = defaultdict(int)
        for ids, freq in ids_freq_table:
            i =  0
            stop = len(ids)-1
            while i<=stop:
                next = i + 1
                pair_counts[(ids[i],ids[next])] += freq
                if ids[i] == ids[next] and next+1<stop and ids[i] == ids[next+1]: #if a pair appears multiple time within one ids
                    i += 2
                else:
                    i = next
        return pair_counts
    
    def merge_batch(self,ids_freq_table, pairs):
        # pairs ---> {(pair1,pair2):idx}
        for ids, freq in ids_freq_table:
            stop = len(ids) - 1
            i = 0
            while i < stop:
                next = i + 1
                token = pairs.get((ids[i], ids[next])) # get id of two consecutive tokens , if they exits as an actual pair in a merged pairs
                if token is not None: 
                    ids[i] = token # replace the two consecutive id with id of pair token
                    del ids[next]
                    last_index -= 1
                i = next

    def train(self, data : list[str], batch_size = 2, max_batch_size = 0, cap_divisor = 2):
        merges = self.merges
        vocab = self.vocab
        batch_count = 0
        curr_vocab_size = len(vocab) + len(self.special_tokens)
        num_merges = self.vocab_size - curr_vocab_size
        merges_remaining = num_merges
        ids = self.texts_to_ids_freq_table(data) 
        if max_batch_size < 1:
            max_batch_size = num_merges

        while merges_remaining > 0:
            seen_first = set() 
            seen_last = set()   
            pairs_to_merge = {}
            stats = self.get_stats(data)
            num_pairs_to_search = min(merges_remaining//cap_divisor, len(vocab), max_batch_size) or 1
            top_pairs = nlargest(num_pairs_to_search, stats, key=stats.get)
            for first, last in top_pairs:  # pairs are (first, last) tuples
                if first in seen_last or last in seen_first:   # unsafe merge
                    seen_first.add(first)
                    seen_last.add(last)
                    continue # skip this pair but keep looking for safe merges in top_pairs
                seen_first.add(first)
                seen_last.add(last)
                pairs_to_merge[(first, last)] = curr_vocab_size
                vocab[curr_vocab_size] = vocab[first] + vocab[last]
                curr_vocab_size += 1
            merges_remaining -= len(pairs_to_merge)
            merges.update(pairs_to_merge)
            batch_count += 1
            if merges_remaining:
                self.merge_batch(ids)



    def decode(self, text_ids):
      #give a list of integers return a python string
      bytes_of_text = [] # this are bytes of text from 
      for idx in text_ids:
            if idx in self.vocab:
                bytes_of_text.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                bytes_of_text.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
      text_bytes = b"".join(bytes_of_text) 
      text = text_bytes.decode("utf-8", errors = "replace")
      return text
    
    def merge(self, ids : list, pair : tuple, idx, len_ids):
        i = 0
        while i + 1 < len_ids:
            next = i + 1
            if ids[i] == pair[0] and ids[next] == pair[1]:
                ids[i] = idx
                del ids[next]
                len_ids -= 1
            i = next
        return len_ids
    
    def _encode_chunk(self, chunk):
        if chunk in self.stop_words:
            return [self.stop_words[chunk]]
        chunk_ids = [*chunk_ids.encode("utf-8")]
        len_chunk = len(chunk_ids)
        while len_chunk >= 2:
            low = 987654321
            for i in range(len_chunk-1):
                current_pair = (chunk_ids[i],chunk_ids[i+1])
                new_val = self.merges.get(current_pair, 987654321)
                if new_val < low:
                    pair = current_pair
                    low = new_val
            if low == 987654321: 
                break   
            idx = self.merges[pair] # get pair id 
            len_chunk = self.merge(chunk_ids, pair, idx, len_chunk) # replace the consectuve pair digits with this single id
        return chunk_ids  
    
    def normal_encode(self,text):
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)
        return ids
    
    def special_encode(self,text,allowed_special):
      """For encoding of special characters"""
      special = {}
      # if all special tokens are allowed by default
      if allowed_special == "all":
        special = self.special_tokens
      # if a specific group of special tokens are allowed from within the group
      elif isinstance(allowed_special,set):
        special = {k:v for k,v in self.special_tokens if k in allowed_special}
      
      elif allowed_special == "none":
        special = {}
      else:
        print(f"{allowed_special} is not a valid input value for the allowed_special parameter")

      if not special:
        return self.normal_encode(text)
      else:
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.normal_encode(part))
        return ids


