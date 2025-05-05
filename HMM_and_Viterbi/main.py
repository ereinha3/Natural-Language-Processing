from collections import Counter, defaultdict
import math
import os


def get_sentences(file_paths):
    sentences = []
    for file_path in file_paths:
        file = open(file_path, 'r')
        buffer = []

        for line in file:
            # EOF
            if not line:
                continue
            # New sentence
            elif line.strip() == '':
                sentences.append(buffer)
                buffer = []
                continue
            # New word
            else:
                tokens = line.strip().split('\t')
                buffer.append(tokens)

    return sentences


def train_hmm_smoothed(train_sents, alpha=0.1):
    """
    Train an HMM with add-alpha (Laplace) smoothing on transitions and emissions,
    and build word->tag dictionary for restricting tag-space.
    Returns log_pi, log_A, log_B, tagset, vocab, word_tag_dict.
    """
    # Count occurrences
    start_counts = Counter()
    trans_counts = defaultdict(Counter)
    emit_counts  = defaultdict(Counter)
    vocab = set()

    for sent in train_sents:
        start_counts[sent[0][1]] += 1
        prev = None
        for w, t in sent:
            vocab.add(w)
            emit_counts[t][w] += 1
            if prev is not None:
                trans_counts[prev][t] += 1
            prev = t

    # Build word to possible tags dict
    word_tag = defaultdict(set)
    for t, ctr in emit_counts.items():
        for w in ctr:
            word_tag[w].add(t)

    tagset = list(emit_counts.keys())
    T = len(tagset)
    V = len(vocab)
    N = len(train_sents)

    # Initial with smoothing and log-probabilities
    log_pi = {}
    for t in tagset:
        log_pi[t] = math.log((start_counts[t] + alpha) / (N + alpha * T))

    # Transition with smoothing and log-probabilities
    log_A = {}
    for u in tagset:
        total = sum(trans_counts[u].values()) + alpha * T
        log_A[u] = {}
        for v in tagset:
            count_uv = trans_counts[u].get(v, 0)
            log_A[u][v] = math.log((count_uv + alpha) / total)

    # Emission with smoothing and log-probabilities
    log_B = {}
    for t in tagset:
        total = sum(emit_counts[t].values()) + alpha * (V + 1)
        log_B[t] = {}
        # known words
        for w, cnt in emit_counts[t].items():
            log_B[t][w] = math.log((cnt + alpha) / total)
        # unknown
        log_B[t]['<UNK>'] = math.log(alpha / total)

    return log_pi, log_A, log_B, tagset, vocab, word_tag

def viterbi_log(words, log_pi, log_A, log_B, tagset, word_tag):
    """
    Viterbi decoding in log-space with:
      - add-alpha smoothed emission probs (with explicit <UNK>)
      - restricted tag-space for seen words
    """
    n = len(words)
    # DP tables
    V = [defaultdict(lambda: -math.inf) for _ in range(n)]
    bp = [{} for _ in range(n)]

    # Initialization
    w0 = words[0]
    possible_tags = word_tag.get(w0, tagset)
    for t in possible_tags:
        emis = log_B[t].get(w0, log_B[t]['<UNK>'])
        V[0][t] = log_pi[t] + emis
        bp[0][t] = None

    # Recursion
    for i in range(1, n):
        w = words[i]
        cur_tags = word_tag.get(w, tagset)
        prev_tags = [t for t in V[i-1] if V[i-1][t] > -math.inf]

        for v in cur_tags:
            emis = log_B[v].get(w, log_B[v]['<UNK>'])
            best_score, best_prev = -math.inf, None
            for u in prev_tags:
                score = V[i-1][u] + log_A[u].get(v, -math.inf) + emis
                if score > best_score:
                    best_score, best_prev = score, u
            V[i][v] = best_score
            bp[i][v] = best_prev

    # Termination & backtrace
    # choose best last tag
    last_tag = max(V[-1], key=V[-1].get)
    tags_seq = [last_tag]
    for i in range(n-1, 0, -1):
        tags_seq.append(bp[i][tags_seq[-1]])
    return list(reversed(tags_seq))


def tag_file(input_words_path, output_pos_path, pi, A, B, tagset, vocab, word_tag):
    with open(input_words_path) as f_in, open(output_pos_path, 'w') as f_out:
        sentence = []
        for line in f_in:
            w = line.strip()
            if not w:
                if sentence:
                    pred_tags = viterbi_log(sentence, pi, A, B, tagset, word_tag)
                    for word, tag in zip(sentence, pred_tags):
                        f_out.write(f"{word}\t{tag}\n")
                    f_out.write("\n")
                    sentence = []
            else:
                sentence.append(w)
        # last sentence if no trailing newline
        if sentence:
            pred_tags = viterbi_log(sentence, pi, A, B, tagset, word_tag)
            for word, tag in zip(sentence, pred_tags):
                f_out.write(f"{word}\t{tag}\n")
            f_out.write("\n")


train_sents = get_sentences(['data/POS_train.pos'])
pi, A, B, tagset, vocab, word_tag = train_hmm_smoothed(train_sents)

tag_file('data/POS_dev.words', 'data/my_dev.pos', pi, A, B, tagset, vocab, word_tag)

os.system("python scorer.py data/POS_dev.pos data/my_dev.pos")

train_and_dev_sents = get_sentences(['data/POS_train.pos', 'data/POS_dev.pos'])
pi, A, B, tagset, vocab, word_tag = train_hmm_smoothed(train_and_dev_sents)

tag_file('data/POS_test.words', 'data/my_test.pos', pi, A, B, tagset, vocab, word_tag)

