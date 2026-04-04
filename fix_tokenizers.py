import pickle

files = [
    'model/rnn_tokenizer.pkl',
    'model/rnn_tokenizer_stress.pkl',
]

for fname in files:
    with open(fname, 'rb') as f:
        tok = pickle.load(f)
    
    plain = {
        'word_index': tok.word_index,
        'num_words':  tok.num_words,
    }
    
    with open(fname, 'wb') as f:
        pickle.dump(plain, f)
    
    vocab_size = len(plain['word_index'])
    print(f'Resaved {fname} as plain dict, vocab={vocab_size}')