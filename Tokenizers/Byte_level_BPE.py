class Tokenizer:
    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}

    def get_stats(self,tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair,0) + 1
        # pair : appearance_count
        return counts
    
    def merge(self,pair, idx,tokens):
        newids = []
        i = 0
        while i<len(tokens):
            if i<len(tokens)-1 and (tokens[i] == pair[0] and tokens[i+1] == pair[1]):
                newids.append(idx)
                i += 2
            else:
                newids.append(tokens[i])
                i += 1
        return newids

    def train(self, tokens):
        current_tokens = tokens 
        idx = 256
        num_merges = self.vocab_size - idx 
        for i in range(num_merges): 
            stats = self.get_stats(current_tokens) 
            top_pair = max(stats, key=stats.get)
            new_idx = idx+i
            current_tokens = self.merge(top_pair,new_idx,current_tokens) # update the list of tokens with its merged version each time
            self.merges[top_pair] = new_idx
    

    
    
    