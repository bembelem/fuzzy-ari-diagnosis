import re, sys
from pathlib import Path
from lxml import etree
from latex2mathml.converter import convert

XSL = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

F = {
 1: r"A=\{(x,\mu_A(x))\mid x\in X\},",
 2: r"\mu_A(x)=\frac{n_A}{n},",
 3: r"\mu_{A\cap B}(x)=\min\left(\mu_A(x),\mu_B(x)\right),\text{  }x\in X.",
 4: r"\mu_{A\cup B}(x)=\max\left(\mu_A(x),\mu_B(x)\right),\text{  }x\in X.",
 5: r"\mu_{\neg A}(x)=1-\mu_A(x),\text{  }x\in X.",
 6: r"\lambda(A\rightarrow B)=\max\left(1-\lambda(A),\lambda(B)\right).",
 7: r"\lambda(A\leftrightarrow B)=\min\left(\max\left(1-\lambda(A),\lambda(B)\right),\max\left(1-\lambda(B),\lambda(A)\right)\right).",
 8: r"\mu=\sum_{i=1}^{n}{w_i\mu_A(x_i)},\text{  }\sum_{i=1}^{n}w_i=1,",
 9: r"\text{ЕСЛИ }x_1\text{ есть }A_1\text{ И }x_2\text{ есть }A_2\text{ И }\ldots\text{ И }x_n\text{ есть }A_n\text{, ТО }y\text{ есть }B,",
 10: r"d_H(A,E)=\frac{1}{n}\sum_{i=1}^{n}{\left|\mu_A(x_i)-\mu_E(x_i)\right|}.",
 11: r"d_E(A,E)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}{\left(\mu_A(x_i)-\mu_E(x_i)\right)^2}}.",
 12: r"S(A,E)=1-d(A,E),",
 13: r"r(A,E)=\frac{\sum_{i=1}^{n}{\left(\mu_A(x_i)-m_A\right)\left(\mu_E(x_i)-m_E\right)}}{\sqrt{\sum_{i=1}^{n}\left(\mu_A(x_i)-m_A\right)^2\cdot\sum_{i=1}^{n}\left(\mu_E(x_i)-m_E\right)^2}},",
 14: r"\mu_T(d)=\max_{s\in S}\text{ }\min\left(\mu_Q(s),\mu_R(s,d)\right).",
 15: r"\mu(x_1)=\min\left(1,\max\left(0,\frac{t-37}{2}\right)\right).",
 16: r"\Delta=S_{(1)}-S_{(2)}.",
 17: r"S(A,E)=1-\sqrt{\frac{\sum_{i}{w_i\left(\mu_A(x_i)-\mu_E(x_i)\right)^2}}{\sum_{i}w_i}},",
 18: r"S'_4=\min\left(S_4,\min\left(1,\text{ }4\cdot\mu_A(x_4)\right)\right),",
 19: r"S'_5=\min\left(S_5,1-\mu_A(x_1)\right),",
 20: r"s=S_3-\max_{k\neq 3}\text{ }S_k,",
}


def q(t): return f"{{{M}}}{t}"
FN = re.compile(r"(min|max)")
def upright(text):
    r = etree.Element(q("r")); rp = etree.SubElement(r, q("rPr")); st = etree.SubElement(rp, q("sty")); st.set(q("val"), "p")
    etree.SubElement(r, q("t")).text = text
    return r
def fix(om):
    # нижние пределы у max/min: sSub -> limLow
    for ss in list(om.iter(q("sSub"))):
        e = ss.find(q("e")); rs = list(e)
        if len(rs) == 1 and rs[0].tag == q("r") and rs[0].findtext(q("t")) in ("min", "max"):
            ll = etree.Element(q("limLow")); le = etree.SubElement(ll, q("e")); le.append(upright(rs[0].findtext(q("t"))))
            lim = etree.SubElement(ll, q("lim"))
            for c in list(ss.find(q("sub"))): lim.append(c)
            ss.getparent().replace(ss, ll)
    # имена функций прямым шрифтом
    for r in list(om.iter(q("r"))):
        if r.find(q("rPr")) is not None: continue
        t = r.findtext(q("t")) or ""
        if not FN.search(t): continue
        parts = [x for x in FN.split(t) if x]
        par = r.getparent(); i = par.index(r); par.remove(r)
        for j, x in enumerate(parts):
            if x in ("min", "max"): n = upright(x)
            else:
                n = etree.Element(q("r")); etree.SubElement(n, q("t")).text = x
            par.insert(i + j, n)
    for t in om.iter(q("t")):
        if t.text and (t.text != t.text.strip()):
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return om

def omml(tex):
    mml = convert(tex)
    res = XSL(etree.fromstring(mml.encode()))
    om = res.getroot()
    if om.tag != f"{{{M}}}oMath":
        om = om.find(f".//{{{M}}}oMath")
    om = fix(om)
    s = etree.tostring(om, encoding="unicode")
    return re.sub(r'\sxmlns:\w+="[^"]*"', "", s)

path = Path(sys.argv[1])
x = path.read_text(encoding="utf8")
done = []
def repl(mt):
    p = mt.group(0)
    if 'w:val="af"' not in p:
        return p
    txt = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))
    n = re.search(r"\((\d+)\)$", txt)
    if not n:
        return p
    k = int(n.group(1))
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", p).group(0)
    done.append(k)
    return (p[:p.index(">") + 1] + ppr + "<w:r><w:tab/></w:r>" + omml(F[k]) +
            f"<w:r><w:tab/><w:t>({k})</w:t></w:r></w:p>")
x = re.sub(r"<w:p[ >](?:(?!</w:p>).)*</w:p>", repl, x, flags=re.S)
# с пределами суммирования в (10) пояснение про индекс больше не нужно
old = "где суммирование ведётся по i от 1 до n. "
assert x.count(old) == 1, x.count(old)
x = x.replace(old, "")
path.write_text(x, encoding="utf8")
print(sorted(done))
