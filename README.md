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

### GPU is optional but worth it

The embedding and reranker models run on CPU by default. If you have a CUDA GPU, reinstall torch for it after `pip install -e .`:

```bash
pip install --force-reinstall torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

`sentence-transformers` picks it up automatically, no config needed. Indexing nanochat takes about **1m38s on a GTX 1050 Ti**; the same job on CPU ran past 50 minutes without finishing.

Pick the CUDA build your card supports. The `cu126` above is deliberate — CUDA 13 dropped Pascal-generation GPUs (anything below compute capability 7.5), so a `cu130` wheel silently won't work on a GTX 10-series card even though the driver advertises CUDA 13.

## Usage

```bash
repochat index https://github.com/karpathy/nanochat.git
repochat repos
repochat status repo_5a5635a5
repochat chat repo_5a5635a5
repochat sources ae_183e701c-S1
```

Indexing prints what it kept and what it skipped:

```
$ repochat index https://github.com/karpathy/nanochat.git

Indexing https://github.com/karpathy/nanochat.git
At commit 92d63d4e8bb4
┌──────────┬───────┐
│ Status   │ Count │
├──────────┼───────┤
│ code     │    30 │
│ test     │     6 │
│ docs     │     4 │
│ config   │     1 │
│ excluded │    12 │
└──────────┴───────┘
Indexed: 310 chunks
Status: READY
Repository ID: repo_5a5635a5
```

Nothing is dropped silently — `status` explains every exclusion after the fact:

```
$ repochat status repo_5a5635a5

State: READY
Commit: 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
Files: code 30, test 6, docs 4, config 1, excluded 12
Chunks: 310

Diagnostics:
  dev/nanochat.png - excluded (unsupported file type)
  uv.lock - excluded (unsupported file type)
  dev/scaling_analysis.ipynb - excluded (unsupported file type)
```

### Questions get routed by shape

A question about one function and a question about how data moves across the app need very different amounts of evidence. A small model labels each question first, and that label sets how wide the rest of the pipeline searches:

| scope | search pool | files kept | chunks to the model |
| --- | --- | --- | --- |
| `symbol` | 20 | 3 | 6 |
| `feature` | 40 | 5 | 8 |
| `repository_flow` | 60 | 8 | 12 |

The difference is visible in the answers. *"What does save_checkpoint do?"* routes to `symbol` and returns six candidates from essentially one file, with one of them cited. *"Trace how base_eval.py loads a checkpoint and evaluates the model end to end"* routes to `repository_flow`, returns seven candidates spanning four files, and cites all seven.

The router only decides how wide to search. It never answers the question and never picks which chunks are relevant — that narrow job is what makes it safe to run on a small, cheap model. If it fails or returns something unparseable, a rule-based classifier takes over.

## What it doesn't do

Worth knowing before you try it on your own code.

**Top-level script code isn't chunked.** Only functions, classes and methods become chunks. A training script that does its real work in top-level statements has that logic missing from the index entirely. This showed up asking how `base_train.py` saves checkpoints: the answer was correct but unhelpful, because the orchestration lives outside any function. The model correctly said it lacked evidence rather than inventing an answer — grounding held, coverage didn't.

**No conversation memory.** Each question is answered independently. "What about the optimizer?" gets reasonable retrieval but no idea what "the" refers to.

**README and config files are catalogued but not chunked**, so they can't inform an answer.

**TypeScript isn't supported.** Only `.py`, `.js`, `.jsx`, `.mjs`, `.cjs`. Worth knowing because plenty of repositories that look like JavaScript are actually TypeScript now.

**The import graph only knows about imports.** If a class receives its dependency through the constructor rather than importing it, no edge exists and graph expansion can't follow that relationship. Imports also don't prove what executes at runtime — the graph is a retrieval aid, not a verifier.

**No benchmark.** There are no recall@k numbers and no ablation table. Retrieval quality has been verified on specific queries, not measured. That's the most useful thing still missing, and the per-answer retrieval trace already stores exactly the per-stage data those metrics would need.

**Re-indexing rebuilds everything.** Content hashes are stored so unchanged chunks could be skipped, but that logic isn't written yet.

## Built with

| | |
| --- | --- |
| Parsing | Python `ast`, Tree-sitter |
| Embeddings | `gte-modernbert-base`, 768-dim, local |
| Keyword search | `bm25s` |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder, local |
| Storage and vector search | PostgreSQL 16 + pgvector |
| Answer generation | Groq, `gpt-oss-120b` |
| Query routing | Groq, `gpt-oss-20b` |
| CLI | Typer + Rich |

Embeddings run locally and generation is hosted, because the volume profiles are opposite: embedding is hundreds of calls per repository, where a free API tier would throttle and a paid one would cost real money, while answering is one call per question and benefits from a far bigger model than fits on a laptop.

Every provider sits behind an interface — `EmbeddingProvider`, `LLMProvider`, `RerankerProvider`, `ScopeClassifier` — so swapping Groq for something else is a config change. The mock implementations are also why the test suite needs no network, no GPU and no database, and finishes in a few seconds.

## Tests

```bash
pytest tests/unit -q      # 117 tests
```

They cover URL validation, file classification, both parsers, chunking and identifier splitting, rank fusion, file scoring, import resolution, score normalization, token budgeting, citation validation and repair, and scope classification.
