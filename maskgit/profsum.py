import sqlite3,sys,re
c=sqlite3.connect(sys.argv[1])
cols=[r[1] for r in c.execute("pragma table_info(kernels)")]
name="name" if "name" in cols else [x for x in cols if "name" in x.lower()][0]
rows=c.execute(f"select {name}, sum(end-start), count(*), avg(end-start) from kernels group by {name} order by 2 desc").fetchall()
tot=sum(r[1] for r in rows)
for n,t,k,a in rows[:22]:
    n=re.sub(r"\s+"," ",n)[:110]
    print(f"{t/tot*100:5.1f}%  {k:6d}  {a/1e3:8.1f}us  {n}")
print("total GPU ms", tot/1e6)
