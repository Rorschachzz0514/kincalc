# -*- coding: utf-8 -*-
"""KinSynth-EN: English synthetic biographies (Wikipedia-biography style, like Re-DocRED) with exact ego-rooted labels.
Same recipe as the Chinese generator: sample a genealogy, render with formulaic sentences, derive labels for the ego by the
calculus. Codes follow the Re-DocRED conversion (gender-neutral where the inventory is: P parent, C child, E spouse, G sibling;
F/M for father/mother when stated). Ego-rooted instances only (the Chinese ablation showed other-rooted instances interfere).
Usage: python gen_synth_en.py --n 2000 --out ../M_模型/sft/SYNEN_train.jsonl [--seed 0]
"""
import os, sys, json, random, argparse
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, "..", "M_模型")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
random.seed(a.seed)
HEADER = ("Below is a document. Taking \"Zeus\" as the central person, list every person named in the document who is a family relative of the central person, "
          "with a relation code read step by step from the central person: F=father M=mother P=parent S=son D=daughter C=child H=husband W=wife E=spouse B=brother Z=sister G=sibling. "
          "Examples: grandfather=FF, parent's sibling=PG, child's spouse=CE, spouse's father=EF.\nOne person per line: name | code\n\n")   # identical to the Re-DocRED prompt header
MALE = "James John Robert William Richard Charles Thomas George Henry Edward Joseph Frederick Arthur Albert Walter Francis Harold Alfred Samuel Louis Ernest Hugh Philip Andrew Peter Simon Nicholas Alexander Patrick Daniel Matthew Benjamin Christopher Jonathan Oliver Stephen Michael David Paul Mark Anthony Victor Leopold Maximilian Ferdinand Otto Karl Friedrich Wilhelm Ludwig Heinrich Johann Pierre Jean Louis Jacques Henri Antoine Carlos Juan Pedro Luis Giovanni Marco Pietro Lorenzo".split()
FEMALE = "Mary Elizabeth Anne Margaret Catherine Jane Eleanor Charlotte Emma Sarah Alice Louise Victoria Harriet Isabella Sophia Caroline Matilda Frances Helen Edith Agnes Beatrice Florence Lucy Rose Clara Dorothy Amelia Julia Maria Anna Teresa Christina Johanna Wilhelmina Augusta Henrietta Marguerite Jeanne Marie Adele Carlotta Giulia Francesca Isabel Juana Ines Leonor".split()
SURN = "Smith Johnson Williams Brown Taylor Davies Wilson Evans Thomas Roberts Walker Wright Robinson Thompson White Hughes Edwards Green Hall Wood Harris Lewis Martin Jackson Clarke Turner Hill Cooper Ward Morris King Baker Harrison Morgan Allen Scott Young Bell Howard Marshall Russell Fisher Mason Parker Grant Spencer Howell Douglas Hamilton Stewart Campbell Murray Gordon Fraser Mueller Schmidt Fischer Weber Wagner Becker Hoffmann Koch Richter Dubois Moreau Laurent Lefebvre Garcia Fernandez Lopez Rossi Russo Ferrari Esposito Bianchi Romano".split()
PLACES = "London Edinburgh Dublin Bristol York Oxford Cambridge Manchester Glasgow Paris Lyon Vienna Prague Berlin Munich Hamburg Zurich Geneva Milan Florence Rome Naples Madrid Seville Lisbon Boston Philadelphia Baltimore Richmond Charleston Montreal Toronto Sydney Melbourne Calcutta Bombay".split() + ["New York", "Cape Town", "The Hague", "St Petersburg"]
PROF = ["politician", "lawyer", "physician", "painter", "composer", "architect", "merchant", "clergyman", "soldier", "naval officer", "judge", "poet", "engineer", "banker", "diplomat", "botanist", "historian", "landowner", "printer", "actor", "sculptor", "astronomer", "surgeon", "professor of mathematics", "member of parliament"]
TITLES = ["Sir", "Lord", "Baron", "Count", "Colonel", "Captain", "Dr", "Rev.", "Admiral", "General", "Duke", "Earl", "Viscount"]

class P:
    def __init__(self, g, surname=None):
        self.g = g; self.first = random.choice(MALE if g else FEMALE); self.sur = surname or random.choice(SURN); self.name = self.first + " " + self.sur
        self.year = None
def sample_family():
    """returns ego, list of (person, code) with codes relative to the ego, and a list of sentences"""
    g = random.random() < 0.65; ego = P(g); rel = []; sents = []
    by = random.randint(1520, 1910); ego.year = by
    place = random.choice(PLACES); prof = random.choice(PROF)
    opening = random.choice(["%s (%d–%d) was an English %s." % (ego.name, by, by + random.randint(38, 85), prof), "%s (born %d in %s) was a %s." % (ego.name, by, place, prof), "%s (%d – %d) was a %s, active mainly in %s." % (ego.name, by, by + random.randint(40, 80), prof, place)])
    sents.append(opening)
    # parents
    fa = P(True, ego.sur); mo = P(False); has_f = random.random() < 0.85; has_m = random.random() < 0.65
    if has_f and has_m:
        sents.append(random.choice(["%s was the %s of %s, a %s, and his wife %s." % (("He" if g else "She"), ("son" if g else "daughter"), fa.name, random.choice(PROF), mo.name), "Born in %s, %s was the %s of %s and %s (née %s)." % (place, ego.first, ("son" if g else "daughter"), fa.name, mo.name, mo.sur), "%s parents were %s and %s." % (("His" if g else "Her"), fa.name, mo.name)]))
        rel += [(fa, "F"), (mo, "M")]
    elif has_f:
        sents.append(random.choice(["%s was the %s of %s, a %s of %s." % (("He" if g else "She"), ("son" if g else "daughter"), fa.name, random.choice(PROF), random.choice(PLACES)), "%s father, %s, was a %s." % (("His" if g else "Her"), fa.name, random.choice(PROF))]))
        rel.append((fa, "F"))
    elif has_m:
        sents.append("%s mother was %s." % (("His" if g else "Her"), mo.name)); rel.append((mo, "M"))
    # grandfather (paternal) sometimes
    if has_f and random.random() < 0.35:
        gf = P(True, ego.sur); sents.append(random.choice(["%s grandfather, %s, had been a %s in %s." % (("His" if g else "Her"), gf.name, random.choice(PROF), random.choice(PLACES)), "The family descended from %s, %s's father, a %s." % (gf.name, fa.name, random.choice(PROF))])); rel.append((gf, "FF"))
    # siblings
    for _ in range(random.choice([0, 0, 1, 1, 2])):
        sg = random.random() < 0.5; sib = P(sg, ego.sur)
        sents.append(random.choice(["%s %s %s was a %s." % (("His" if g else "Her"), ("brother" if sg else "sister"), sib.name, random.choice(PROF)), "%s was the %s of %s." % (ego.first, ("brother" if g else "sister"), sib.name), "%s had %s, %s." % (("He" if g else "She"), ("a brother" if sg else "a sister"), sib.name)]))
        rel.append((sib, "B" if sg else "Z"))
    # spouse + children
    if random.random() < 0.8:
        sp = P(not g); sp.sur = random.choice(SURN) if not g else sp.sur
        my = by + random.randint(20, 35)
        wf = P(True, sp.sur) if (g and random.random() < 0.4) else None; wf_txt = (", daughter of %s" % wf.name) if wf else ""
        sents.append(random.choice(["In %d %s married %s%s." % (my, ("he" if g else "she"), sp.name, wf_txt), "%s married %s in %d; the couple settled in %s." % (ego.first, sp.name, my, random.choice(PLACES)), "%s wife was %s." % ("His", sp.name) if g else "%s was the wife of %s." % (ego.first, sp.name)]))
        rel.append((sp, "H" if not g else "W"))
        if wf and "daughter of" in sents[-1]: rel.append((wf, "WF"))
        kids = []
        for _ in range(random.choice([0, 1, 2, 2, 3])):
            kg = random.random() < 0.5; k = P(kg, ego.sur if g else sp.sur); kids.append(k); rel.append((k, "S" if kg else "D"))
        if kids:
            if len(kids) == 1: sents.append(random.choice(["They had one %s, %s." % (("son" if kids[0].g else "daughter"), kids[0].name), "%s %s %s became a %s." % (("His" if g else "Her"), ("son" if kids[0].g else "daughter"), kids[0].name, random.choice(PROF))]))
            else: sents.append(random.choice(["Their children included %s and %s." % (", ".join(k.name for k in kids[:-1]), kids[-1].name), "They had %d children, among them %s and %s." % (len(kids) + random.randint(0, 2), ", ".join(k.name for k in kids[:-1]), kids[-1].name)]))
            if random.random() < 0.3 and kids[0].g != None:
                sil = P(not kids[0].g); sents.append("%s %s married %s." % (("His" if g else "Her"), ("son" if kids[0].g else "daughter"), sil.name) if False else "%s later married %s." % (kids[0].name, sil.name)); rel.append((sil, ("S" if kids[0].g else "D") + ("W" if kids[0].g else "H")))
    # filler sentences about career
    fill = ["%s was educated at %s and entered the %s profession in %d." % (("He" if g else "She"), random.choice(["Eton", "Harrow", "Westminster School", "Trinity College", "the Sorbonne", "Heidelberg", "a grammar school in " + random.choice(PLACES)]), random.choice(["legal", "medical", "naval", "clerical", "diplomatic"]), by + random.randint(18, 26)),
            "%s works were exhibited in %s and %s." % (("His" if g else "Her"), random.choice(PLACES), random.choice(PLACES)), "%s served as %s of %s from %d to %d." % (("He" if g else "She"), random.choice(["mayor", "sheriff", "governor", "dean", "director"]), random.choice(PLACES), by + 40, by + 45),
            "%s died in %s and was buried at %s." % (("He" if g else "She"), random.choice(PLACES), random.choice(["St Mary's", "the cathedral", "the family vault", "Highgate Cemetery", "Père Lachaise"]))]
    random.shuffle(fill); sents = [sents[0]] + fill[:1] + sents[1:] + fill[1:3]
    return ego, rel, sents
NEU = {"S": "C", "D": "C", "H": "E", "W": "E", "B": "G", "Z": "G"}
def gender_neutral(code):
    """Re-DocRED keeps father/mother but has only child/spouse/sibling, so KinSynth-EN labels use the same inventory"""
    return "".join(NEU.get(ch, ch) for ch in code)
out = []
for i in range(a.n):
    ego, rel, sents = sample_family()
    text = " ".join(sents)
    rel = [(p, gender_neutral(c)) for p, c in rel]
    target = "\n".join("%s | %s" % (p.name, c) for p, c in rel)
    out.append({"id": "SYNEN%d" % i, "root": ego.name, "g_root": 1 if ego.g else 0, "gold": [{"name": p.name, "aliases": [p.name], "path": c} for p, c in rel], "maxhop": 0, "prompt": HEADER.replace("\"Zeus\"", "\"%s\"" % ego.name) + "Document:\n" + text, "target": target})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("wrote", len(out), "docs; mean relatives %.2f" % (sum(len(d["gold"]) for d in out) / len(out))); print(out[0]["prompt"][-500:]); print(out[0]["target"])
