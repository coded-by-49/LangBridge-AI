import regex as re
from collections import Counter
from collections import defaultdict
import unicodedata
naija_eng_pattern = re.compile(r"""
(
    \s+
  | \p{L}[\p{L}\p{M}'’\-]*(?!\p{M})



  | \p{N}+
  | [\p{P}\p{S}]+
  | .
)
""", re.UNICODE | re.VERBOSE)

# self.special_tokens = ("<|startoftext|>","<|endoftext|>","<|endofprompt|>") #come back to this
def replace_control_characters(self, s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

"""Return a cleaned version of each string"""
def render_token(self, t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = self.replace_control_characters(s)
    return s



class RegrexBpeTokenizer:
    def __init__(self,vocab_size,stop_list_size = 100):
        assert vocab_size >= 256
        self.vocab_size = vocab_size
        self.stop_words = None
        self.stop_list_size = stop_list_size
        self.split_pattern = naija_eng_pattern
        self.compiled_pattern = re.compile(self.split_pattern)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        

    def train(self, list_of_text, batch_size):
        start = 0
        end = batch_size
        while start <= 34350000:
            text= " ".join(list_of_text[start:end])
            text_chunks = re.findall(self.compiled_pattern, text)
            ids = [list(word.encode("utf-8")) for word in text_chunks]
            current_ids = ids # copying out ids
            idx = 256
            num_merges = self.vocab_size - idx
            for i in range(num_merges):
                stats = Counter()
                # look through each pair and return the one with the highest frequency
                for chunk_ids in current_ids:
                    self.get_stats(chunk_ids, stats) 
                if stats:
                    top_pair = max(stats, key=stats.get)
                else:
                    print("No more pairs to merge. Stopping training.")
                    break
                new_idx = idx+i
                current_ids = [self.merge(top_pair,new_idx,chunk_ids) for chunk_ids in current_ids]  # update the token_ids with the new found pairs
                self.merges[top_pair] = new_idx
                self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            end += batch_size
            start += batch_size
        return self.vocab,self.merges
        

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
    
    def get_stats (ids_freq_table):
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
    
    def merge_batch(ids_freq_table, pairs):
        for ids, freq in ids_freq_table:
            stop = len(ids) - 1
            i = 0
            while i < stop:
                next = i + 1
                token = pairs.get((ids[i], ids[next]))
                if token is not None:
                    ids[i] = token
                    del ids[next]
                    last_index -= 1
                i = next

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
    
    #!!!!1
    def _encode_chunk(self, chunks):
        ids = list(chunks)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged anymore
            idx = self.merges[pair]
            ids = self.merge(pair, idx, ids)
        return ids
    
    def normal_encode(self,text):
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
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


