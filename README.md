# RepoChat

Ask questions about a GitHub repository and get answers that cite the exact file and line range they came from.

```
You: What does save_checkpoint do?

RepoChat > save_checkpoint persists a training snapshot to disk. Only rank 0
writes the model parameters, to a file named model_<step>.pt, along with a
meta_<step>.json holding the training metadata [S1]. Optimizer state is
sharded across ranks, so every rank writes its own optim_<step>_rank<rank>.pt
when optimizer_data is provided [S1].

Sources
  [S1] ae_183e701c-S1  nanochat/checkpoint_manager.py:41-58  save_checkpoint
  [S2] (unused)        nanochat/checkpoint_manager.py:76-114 build_model
  [S3] (unused)        nanochat/tokenizer.py:132-138         RustBPETokenizer.save
```

Every `[S1]` resolves to real code. Later, from any terminal:

```
$ repochat sources ae_183e701c-S1

nanochat/checkpoint_manager.py:41-58 | save_checkpoint

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        ...

https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/checkpoint_manager.py#L41-L58
```

The permalink is pinned to the commit that was indexed, so it keeps pointing at the same lines even after the branch moves on.

## Why not just paste the code into ChatGPT

For a single file, you should. It stops working on a real repository for three reasons:

**It doesn't fit.** A few hundred files is far more text than any context window holds, so most of it gets dropped arbitrarily.

**Structure is lost.** Whatever does fit arrives as one undifferentiated wall of text. Nothing tells the model that "the login flow" means three specific files, three folders apart, that call each other in a particular order.

**It can't check itself.** A normal chat answers confidently when it's wrong, because it has no way to verify a claim against the actual code.

RepoChat is three countermeasures to those: retrieve only what's relevant, organize by syntax and file relationships rather than raw text, and force every claim to point at evidence that gets validated before you see it.

## How it works

Two halves that never touch each other except through the database.

**Indexing** runs once per commit. It clones a read-only copy at an exact SHA, decides which files are worth reading, parses each one into functions and classes, and turns each symbol into a separately retrievable chunk with its real line numbers attached. It also records which files import which.

**Answering** runs per question. It searches, narrows, and only then generates.

```
question
   ↓
scope router        narrow lookup or cross-file trace? sets how wide to search
   ↓
BM25  +  vectors    keyword search and semantic search, in parallel
   ↓
rank fusion         merge the two ranked lists
   ↓
diversify           spread picks across files instead of piling onto one
   ↓
graph expansion     pull in directly-connected files, both directions
   ↓
rerank              cross-encoder scores question and chunk together
   ↓
token budget        fit the winners into the context window
   ↓
generate            answer only from these sources, cite every claim
   ↓
verify citations    repair once, then abstain — never silently strip
```

### Chunks are symbols, not text windows

Most RAG splits documents into fixed token windows. For code that breaks things: it cuts functions in half, loses the class a method belongs to, and can't cite anything more precise than "chunk 47".

RepoChat cuts along boundaries the language itself defines, using Python's `ast` module and Tree-sitter for JavaScript. A function is one chunk. A method is one chunk, tagged with its parent class. Every chunk carries `start_line` and `end_line` straight from the parser, which is what makes `checkpoint_manager.py:41-58` possible.

Each chunk is stored twice over: the untouched original source for display and citations, and a metadata-enriched version that actually gets embedded:

```
[repository: karpathy/nanochat] [path: nanochat/checkpoint_manager.py] [language: python]
[symbol: save_checkpoint] [kind: function] [lines: 41-58]
[parent: module] [signature: def save_checkpoint(checkpoint_dir, step, ...):]
SOURCE
def save_checkpoint(checkpoint_dir, step, model_data, ...):
    ...
```

Roughly two-thirds of what gets embedded is metadata. A bare function body embeds to something nearly meaningless; with the header, the same vector can be found by the symbol name, by the file, or by a description of what it does.

### Why keyword search as well as embeddings

They fail in opposite directions, which is easy to show. Searching this index for `print0`:

```
dense only:  print_banner, print0, test_happy_path, ...
hybrid:      print0, print_banner, main, ...
```

Dense retrieval put a decoy first. `print_banner` and `print0` are semantically close — both are "things that print" — and the embedding couldn't separate them. BM25's exact token match fixed it immediately.

It goes the other way too: BM25 alone will never connect "how are passwords hashed" to a function called `derive_key`, because they share no tokens at all. Hybrid covers both blind spots, and the two ranked lists are merged by rank rather than score, since a BM25 score and a cosine distance aren't comparable numbers.

### Citations get verified, not trusted

After the model answers, every `[S#]` is checked against the sources it was actually given. If one doesn't exist, the model gets one chance to correct it. If it still can't, the claim is marked as unsupported rather than having its citation quietly deleted — a silently stripped citation leaves a sentence that still looks sourced.

This is also why there's no token streaming. Once an invalid citation has been printed to the terminal, there's nothing left to repair. Buffer, validate, then display.

## Setup

Requires Python 3.11+, Docker, and a free [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/shaulcifal/Repochat.git
cd Repochat

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"

docker compose -f infra/docker-compose.yml up -d

cp .env.example .env            # then fill in GROQ_API_KEY
```

Everything else runs locally and costs nothing: embeddings and reranking are local models, and Postgres is the container you just started. Only answer generation calls out to Groq's free tier.

### Questions get routed by shape

A question about one function and a question about how data moves across the app need very different amounts of evidence. A small model labels each question first, and that label sets how wide the rest of the pipeline searches:

| scope | search pool | files kept | chunks to the model |
| --- | --- | --- | --- |
| `symbol` | 20 | 3 | 6 |
| `feature` | 40 | 5 | 8 |
| `repository_flow` | 60 | 8 | 12 |

The difference is visible in the answers. *"What does save_checkpoint do?"* routes to `symbol` and returns six candidates from essentially one file, with one of them cited. *"Trace how base_eval.py loads a checkpoint and evaluates the model end to end"* routes to `repository_flow`, returns seven candidates spanning four files, and cites all seven.

The router only decides how wide to search. It never answers the question and never picks which chunks are relevant — that narrow job is what makes it safe to run on a small, cheap model. If it fails or returns something unparseable, a rule-based classifier takes over.


