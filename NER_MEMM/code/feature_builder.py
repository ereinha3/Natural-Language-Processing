import sys

# Below are the imported files from the course website

# This provides a list of surnames
person_names = set()
with open('maxent/dist.all.last.txt') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            person_names.add(parts[0].lower())

# This provides a list of the largest cities
cities = set()
with open('maxent/LargestCity.txt') as f:
    for line in f:
        city = line.strip()
        if city:
            cities.add(city.lower())

# This provides a list of the most common words in the Brown corpus
brown_names = set()
brown_freq = {}
with open('maxent/brown-c1000-freq1.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            _, name, freq = parts[:3]
            lname = name.lower()
            brown_names.add(lname)
            try:
                brown_freq[lname] = int(freq)
            except ValueError:
                brown_freq[lname] = 0

# This function converts a word to a shape pattern (found something on stack overflow that suggested this)
def word_shape(w):
    return ''.join(
        'X' if c.isupper() else
        'x' if c.islower() else
        'd' if c.isdigit() else
        c
        for c in w
    )

# This function builds feature vectors for each token
def extract_features(tokens, poses, chunks, tags=None):
    lines = []
    prev_tag = 'B-ORG'  # needed for first word appearance

    for i, w in enumerate(tokens):
        feats = []
        lw = w.lower()

        # # Adds word to features
        # feats.append(f"WORD={w}")
        # Adds lowercase version of word to features
        feats.append(f"LOWER={lw}")

        # Is first character uppercase
        feats.append(f"INIT_CAP={int(w[0].isupper())}")
        # Is entire word uppercase
        feats.append(f"ALL_CAP={int(w.isupper())}")
        # Does word contain any digits
        feats.append(f"HAS_DIGIT={int(any(c.isdigit() for c in w))}")
        # Does word contain a hyphen
        feats.append(f"HAS_HYPHEN={int('-' in w)}")
        # Converts word to a shape pattern
        feats.append(f"SHAPE={word_shape(w)}")

        # # Length of the word
        # length = len(w)
        # if length >= 10:
        #     bin_lbl = ">=10"
        # elif length >= 5:
        #     bin_lbl = "5-9"
        # else:
        #     bin_lbl = str(length)
        # feats.append(f"LENGTH_BIN={bin_lbl}")

        # Adds prefix/suffix to features
        for j in (1, 2, 3):
            if len(w) >= j:
                feats.append(f"PREFIX{j}={lw[:j]}")
                feats.append(f"SUFFIX{j}={lw[-j:]}")

        # Adds POS and chunk to features
        feats.append(f"POS={poses[i]}")
        feats.append(f"CHUNK={chunks[i]}")

        # # Adds features for whether the word is a person, city, or common word
        feats.append(f"IN_PERSON={int(lw in person_names)}")
        feats.append(f"IN_CITY={int(lw in cities)}")
        feats.append(f"IN_BROWN={int(lw in brown_names)}")
        freq = brown_freq.get(lw, 0)
        feats.append(f"BROWN_FREQ={freq}")
        if w.endswith(('Inc', 'Ltd', 'Corp', 'University')):
            feats.append("ORG_SUFFIX=1")

        # Adds previous tag to features
        tag_feature = prev_tag if tags else "@@"
        feats.append(f"PREV_TAG={tag_feature}")

        # Adds previous word, POS, and POS bigram to features
        if i > 0:
            feats.append(f"PREV_WORD={tokens[i-1].lower()}")
            feats.append(f"PREV_POS={poses[i-1]}")
            feats.append(f"POS_BIGRAM={poses[i-1]}_{poses[i]}")
            feats.append(f"TAG_PREV_POS={tag_feature}_{poses[i-1]}")
        if i < len(tokens) - 1:
            feats.append(f"NEXT_WORD={tokens[i+1].lower()}")
            feats.append(f"NEXT_POS={poses[i+1]}")

        # Combines features into a single line
        gold = tags[i] if tags else None
        parts = [w] + feats + ([gold] if gold else [])
        lines.append("\t".join(map(str, parts)))

        # Prepares previous tag for next token
        prev_tag = gold if tags else "@@"

    return lines

# Training data reader
def read_pos_chunk_name(file_path):
    toks, poses, chks, tags = [], [], [], []
    for line in open(file_path):
        if not line.strip():
            yield toks, poses, chks, tags
            toks, poses, chks, tags = [], [], [], []
            continue
        w, p, c, t = line.strip().split()
        toks.append(w); poses.append(p); chks.append(c); tags.append(t)
    if toks:
        yield toks, poses, chks, tags

# Dev/test data reader
def read_pos_chunk(file_path):
    toks, poses, chks = [], [], []
    for line in open(file_path):
        if not line.strip():
            yield toks, poses, chks
            toks, poses, chks = [], [], []
            continue
        w, p, c = line.strip().split()
        toks.append(w); poses.append(p); chks.append(c)
    if toks:
        yield toks, poses, chks


if __name__ == "__main__":
    mode = sys.argv[1]   # train or literally whatever you want
    inp  = sys.argv[2]   # train.pos-chunk-name or dev.pos-chunk or test.pos-chunk
    out  = sys.argv[3]   # train.feat or dev.feat or test.feat

    with open(out, 'w') as fout:
        # Process training data
        if mode == 'train':
            for toks, poses, chks, tags in read_pos_chunk_name(inp):
                for line in extract_features(toks, poses, chks, tags):
                    fout.write(line + "\n")
                fout.write("\n")
        # Process dev/test data
        else:
            for toks, poses, chks in read_pos_chunk(inp):
                for line in extract_features(toks, poses, chks):
                    fout.write(line + "\n")
                fout.write("\n")
