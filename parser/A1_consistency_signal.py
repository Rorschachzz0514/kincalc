import json,collections
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
st=collections.defaultdict(collections.Counter); doc=collections.defaultdict(collections.Counter)
for f in ('runs/X0/xfit.jsonl','runs/X1/xfit.jsonl'):
    for l in open(f,encoding='utf-8'):
        d=json.loads(l); allok=True; nm=0; anyc=False; nsup=0
        for n,p in d['direct'].items():
            gm=[g['path'] for g in d['gold'] if same(n,g['name'])]
            s,c=d['support'].get(n,0),d['conflict'].get(n,0)
            b='印证>=2,无否定' if s>=2 and c==0 else '印证1,无否定' if s==1 and c==0 else '有否定' if c>0 else '无视角覆盖'
            anyc|=c>0; nsup+= (s>=1 and c==0)
            if gm: st[b]['n']+=1; st[b]['ok']+=p in gm; nm+=1; allok&= p in gm
            else: st[b]['unmatched']+=1
        if nm:
            k='文档:无否定且>=2人被印证' if (not anyc and nsup>=2) else '文档:无否定' if not anyc else '文档:有否定'
            doc[k]['n']+=1; doc[k]['allok']+=allok
print('人物级 (交叉拟合模型在另一半训练集上; 只统计能对上金标的人):')
for b,c in sorted(st.items()): print('  %-14s 对上金标 %5d 路径正确 %.1f%%  (另有金标外 %d 人)'%(b,c['n'],c['ok']/max(1,c['n'])*100,c['unmatched']))
print('文档级 (该篇所有对上金标的人路径全对的比例):')
for b,c in sorted(doc.items()): print('  %-22s %5d 篇  全对 %.1f%%'%(b,c['n'],c['allok']/c['n']*100))
