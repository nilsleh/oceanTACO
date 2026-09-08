"""Sweep tutorial notebook prose for AI-writing patterns.

Two nets: the Patina list (regression net for round 1) and the four
signatures that prose-writing-style targets (S1-S4). Run from the repo root.
"""
import collections
import json
import re
import sys

NBS = ["ml_dataset", "data_retrieval_workflows",
       "spatio_temporal_query_generation", "ml_configuration_cookbook"]

PATINA = {
 "P7 AI vocab": r'\b(delve|tapestry|multifaceted|leverage|synergy|realm|seamless|crucial|pivotal|underscore\w*|showcas\w+)\b',
 "P8 copula avoidance": r'\b(serves? as|acts? as|functions? as)\b',
 "P9 negative parallelism": r'\bnot (just|merely|only)\b[^.]*\bbut\b',
 "P3 -ing chain": r',\s+(showcasing|highlighting|underscoring|reflecting|demonstrating|emphasizing|illustrating)\b',
 "P22 filler": r"\b(it('s| is) (important|worth) (to note|noting)|due to the fact that|it should be noted|as we can see)\b",
 "P31 conclusion signal": r'^\s*(In conclusion|Ultimately|In summary|To summarize|Overall|In short)\b',
 "P35 throat-clearing": r"\b(here'?s the thing|let me be clear|that said,|at the end of the day)\b",
 "P36 faux-insight": r"\b(what nobody tells you|most people (get this wrong|don'?t)|the (secret|trick) is)\b",
 "P24 generic positive": r'\b(bright future|exciting times|poised (for|to)|powerful tool)\b',
 "P34 false agency": r'\b(the data (tells|says)|the (results?|numbers?) (tell|speak|reveal))\b',
 "P37 colon reveal": r'^[A-Z][A-Za-z ]{2,28}:\s+[A-Z]',
 "P12 false range": r'\bfrom [a-z]+ to [a-z]+\b',
 "P32 bare comparative": r'\bmore (efficient|comprehensive|robust|powerful|flexible|natural)\b',
 "P13 em dash": r'—',
 "P17 emoji": r'[\U0001F300-\U0001FAFF]',
 "banned: antithesis": r'\b(is not|are not|rather than) [^.,]{2,40}, (it|they|but) (is|are)\b',
 "banned: what X looks like": r'what .{1,30} looks like',
 "banned: X decides the Y": r'\b\w+ decides (the|what|which)\b',
 "banned: semicolon": r';',
}

SIGNATURES = {
 "S3 cleft emphasis": r'\b(is|are|was) what [a-z]',
 "S4 self-grading": r'\b(this notebook (walks|covers|shows)|on purpose|the honest|punishes)\b',
}


def prose(nb):
    cells = json.load(open(f'docs/tutorials/{nb}.ipynb'))['cells']
    return '\n\n'.join(''.join(c['source']) for c in cells
                       if c['cell_type'] == 'markdown')


def closers(text):
    """Paragraph-final short sentences carrying no number (signature S2)."""
    out = []
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if para.startswith(('#', '|', '-', '*')) or re.match(r'^\d+\.', para):
            continue
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
        if sents and 3 <= len(sents[-1].split()) <= 13 and not re.search(r'\d', sents[-1]):
            out.append(' '.join(sents[-1].split()))
    return out


def main():
    total = collections.Counter()
    shaped = 0
    for nb in NBS:
        text = prose(nb)
        hits = {k: len(re.findall(p, text, re.IGNORECASE | re.MULTILINE))
                for k, p in {**PATINA, **SIGNATURES}.items()}
        hits = {k: v for k, v in hits.items() if v}
        cl = closers(text)
        shaped += len(cl)
        total.update(hits)
        print(f"\n== {nb} ({len(text.split())} words) ==")
        for k, v in sorted(hits.items(), key=lambda x: -x[1]):
            print(f"   {v:3d}  {k}")
        for c in cl:
            print(f"        S2 closer: {c[:72]}")
    print("\n===== TOTAL =====")
    for k, v in total.most_common():
        print(f"   {v:3d}  {k}")
    print(f"   {shaped:3d}  S2 shaped closers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
