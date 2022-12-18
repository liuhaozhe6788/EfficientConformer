import re, collections
import matplotlib.pyplot as plt


def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, "r", encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]]+=freq
    return pairs

# def get_tokens(vocab):
#     tokens = collections.defaultdict(int)
#     for word, freq in vocab.items():
#         word_tokens = word.split()
#         for token in word_tokens:
#             tokens[token] += freq
#     return tokens

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
    for word in v_in:
        w_out =re.sub(p,''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def measure_token_length(token):
    if token[-4: ] == '</w>':
        return len(token[: -4]) + 1  # blank space
    else:
        return len(token)
    
def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]
    
    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0

        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position: substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens


if __name__ == "__main__":
    # vocab = {
    #     'l o w </w>':5, 
    #     'l o w e r </w>':2, 
    #     'n e w e s t </w>':6, 
    #     'w i d e s t </w>':3, 
    # }

    vocab = get_vocab("pg16457.txt")
    print('==========')
    print('Tokens Before BPE')
    tokens, vocab_tokenization = get_tokens_from_vocab(vocab)
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')

    num_merges = 1000000
    token_lens = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        # print('Iter: {}'.format(i))
        # print('Best pair: {}'.format(best))
        tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
        # print('Tokens: {}'.format(tokens))
        # print('Number of tokens: {}'.format(len(tokens)))
        token_lens.append(len(tokens_frequencies))
        # print('==========')
    plt.plot(token_lens)
    plt.savefig("token_lens.png", dpi=600)

sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

print(sorted_tokens)
 
while True:
    word_given = input("please input a string:")

    print('Tokenizing word: {}...'.format(word_given))
    if word_given in vocab_tokenization:
        print('Tokenization of the known word:')
        print(vocab_tokenization[word_given])
        print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))
    else:
        print('Tokenizing of the unknown word:')
        print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))


