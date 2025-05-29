import sys

# load gazetteers
person_names = set(open('dist.all.last.txt').read().split())
cities        = set(open('LargestCity.txt').read().split())

def word_shape(w):
    return ''.join('X' if c.isupper() else 'x' if c.islower()
                   else 'd' if c.isdigit() else c for c in w)

def extract_features(tokens, poses, chunks, tags=None):
    feats, lines = [], []
    prev_tag = 'B-ORG'  # dummy for first token in training
    for i, w in enumerate(tokens):
        f = []
        f.append(f"WORD={w}")
        f.append(f"LOWER={w.lower()}")
        f.append(f"CAP={w[0].isupper()}")
        f.append(f"SHAPE={word_shape(w)}")
        f.append(f"POS={poses[i]}")
        f.append(f"CHUNK={chunks[i]}")
        f.append(f"PREV_TAG={prev_tag}")

        # context
        if i > 0:
            f.append(f"PREV_WORD={tokens[i-1].lower()}")
            f.append(f"PREV_POS={poses[i-1]}")
        if i < len(tokens)-1:
            f.append(f"NEXT_WORD={tokens[i+1].lower()}")
            f.append(f"NEXT_POS={poses[i+1]}")

        # gazetteers
        f.append(f"IN_PERSON={w.lower() in person_names}")
        f.append(f"IN_CITY={w.lower() in cities}")
        if w.endswith(('Inc','Ltd','Corp','University')):
            f.append("ORG_SUFFIX=1")

        tag = tags[i] if tags else None
        line = "\t".join([w] + f + ([tag] if tag else []))
        lines.append(line)

        # for training, gold tag; for dev/test, MEtag will replace PREV_TAG
        prev_tag = tag if tags else "@@"
    return lines

def read_pos_chunk_name(file_path):
    toks, poses, chks, tags = [], [], [], []
    for line in open(file_path):
        if not line.strip():
            yield toks, poses, chks, tags
            toks, poses, chks, tags = [], [], [], []
            continue
        w, p, c, t = line.strip().split()
        toks.append(w); poses.append(p); chks.append(c); tags.append(t)
    if toks: yield toks, poses, chks, tags

def read_pos_chunk(file_path):
    toks, poses, chks = [], [], []
    for line in open(file_path):
        if not line.strip():
            yield toks, poses, chks
            toks, poses, chks = [], [], []
            continue
        w, p, c = line.strip().split()
        toks.append(w); poses.append(p); chks.append(c)
    if toks: yield toks, poses, chks

if __name__ == "__main__":
    mode = sys.argv[1]   # "train" or "dev" or "test"
    inp  = sys.argv[2]   # e.g. train.pos-chunk-name or dev.pos-chunk
    out  = sys.argv[3]   # e.g. train.feat or dev.feat

    with open(out, 'w') as fout:
        if mode == 'train':
            for toks, poses, chks, tags in read_pos_chunk_name(inp):
                for line in extract_features(toks, poses, chks, tags):
                    fout.write(line + "\n")
                fout.write("\n")
        else:
            for toks, poses, chks in read_pos_chunk(inp):
                for line in extract_features(toks, poses, chks):
                    fout.write(line + "\n")
                fout.write("\n")
