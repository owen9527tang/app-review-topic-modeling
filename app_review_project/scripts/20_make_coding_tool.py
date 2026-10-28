"""P4 加固：把两份盲编码表打包成本地 HTML 编码工具。

用法：python scripts/20_make_coding_tool.py
产出：data/coding_tool_A.html（你本人）、data/coding_tool_B.html（交给第二人）

离线可用：双击打开，进度自动存 localStorage，点「导出 CSV」即可覆盖对应编码表。
评论正文只写入这两个本地文件，已被 .gitignore 排除。

刻意不显示星级、机械基线判定与字数——任何一条都会给编码者锚点。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATES = DATA / "topic_topwords.csv"  # 只为确认工作目录正确，不读内容

THEMES = ["fee", "doctor", "delivery", "prescription", "appointment", "tech",
          "privacy", "service", "promotion", "goods", "content", "other"]
THEME_CN = {"fee": "收费/退款", "doctor": "医生质量", "delivery": "药品配送",
            "prescription": "处方/用药", "appointment": "预约/挂号", "tech": "技术/账号",
            "privacy": "隐私/安全", "service": "客服/售后", "promotion": "活动/营销",
            "goods": "商品/实物", "content": "内容/推送", "other": "其他/空泛"}

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>编码工具 __SHEET__</title>
<style>
:root{--bg:#fbfaf7;--fg:#1c1c1a;--dim:#6b6b66;--line:#e2ded6;--accent:#8a5a2b;--ok:#2f6b46}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:"Noto Serif SC","Songti SC",SimSun,serif}
header{position:sticky;top:0;background:rgba(251,250,247,.96);border-bottom:1px solid var(--line);
padding:12px 26px;display:flex;align-items:center;gap:16px;z-index:5;font-size:14px}
header .p{font-variant-numeric:tabular-nums}
header .sp{margin-left:auto;display:flex;gap:9px}
.btn{border:1px solid var(--line);background:#fff;padding:6px 13px;border-radius:6px;font-size:13.5px;
cursor:pointer;font-family:inherit;color:var(--fg)}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:960px;margin:0 auto;padding:30px 26px 130px}
.meta{font-size:12.5px;color:var(--dim);margin-bottom:9px}
.content{background:#fff;border:1px solid var(--line);border-radius:10px;padding:24px 26px;
font-size:20px;line-height:1.75;white-space:pre-wrap;word-break:break-word;min-height:140px}
h3{font-size:12.5px;letter-spacing:.12em;color:var(--dim);font-weight:400;margin:30px 0 10px}
.opts{display:grid;grid-template-columns:repeat(3,1fr);gap:11px}
.opt{text-align:left;border:1px solid var(--line);background:#fff;border-radius:8px;padding:13px 15px;
cursor:pointer;line-height:1.45}
.opt b{display:block;font-size:16px;margin-bottom:3px}
.opt span{font-size:12px;color:var(--dim)}
.opt.on{border-color:var(--accent);background:#fdf6ef}.opt.on b{color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 14px;font-size:13.5px;
cursor:pointer;font-family:inherit;color:var(--fg)}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
textarea{width:100%;min-height:56px;margin-top:8px;padding:9px 11px;font-family:inherit;font-size:13.5px;
border:1px solid var(--line);border-radius:8px;background:#fff;resize:vertical}
nav{position:fixed;bottom:0;left:0;right:0;background:rgba(251,250,247,.97);border-top:1px solid var(--line);
padding:11px 26px;display:flex;gap:11px;align-items:center}
.hint{font-size:12px;color:var(--dim);margin-left:auto}
</style></head><body>
<header><b>编码工具 · __SHEET__ 表</b><span class="p" id="prog">0 / 100</span>
<span class="sp"><button class="btn" id="export">导出 CSV</button></span></header>
<main>
  <div class="meta" id="meta"></div>
  <div class="content" id="content"></div>
  <h3>文本具体性 spec</h3>
  <div class="opts" id="spec"></div>
  <h3>主议题 primary</h3>
  <div class="chips" id="primary"></div>
  <h3>备注 note（可选）</h3>
  <textarea id="note" placeholder="边界判例写在这里，便于事后核对规则"></textarea>
</main>
<nav>
  <button class="btn" id="prev">上一条</button>
  <button class="btn primary" id="next">下一条</button>
  <span class="hint">快捷键：0 / 1 / 2 定具体性，← → 翻页，Enter 下一条</span>
</nav>
<script>
const ROWS = __ROWS__;
const SPECS = [[0,"0 无具体对象","只有情绪或空泛夸奖：垃圾、太好了"],["1","1 有对象无细节","能说清在谈什么，但没有可核对的量"],["2","2 可核实细节","金额、次数、时长、具体功能名、症状/药品、操作步骤"]];
const THEMES = __THEMES__;
const THEME_CN = __THEME_CN__;
const KEY = __KEY__;
const S = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
ROWS.forEach(r => { if(!S[r.review_id]) S[r.review_id] = {spec:"",primary:"",note:""}; });

function save(){ localStorage.setItem(KEY, JSON.stringify(S)); }
function sync(){
  const r = ROWS[i], c = S[r.review_id];
  document.getElementById("meta").textContent = (i+1)+" / "+ROWS.length+" · "+r.app_name+" · ID "+r.review_id;
  document.getElementById("content").textContent = r.content;
  document.getElementById("note").value = c.note || "";
  document.getElementById("spec").innerHTML = SPECS.map(([v,t,d])=>
    '<button class="opt '+(c.spec===String(v)?"on":"")+'" data-v="'+v+'"><b>'+t+'</b><span>'+d+'</span></button>').join("");
  document.getElementById("primary").innerHTML = THEMES.map(k=>
    '<button class="chip '+(c.primary===k?"on":"")+'" data-k="'+k+'">'+THEME_CN[k]+'</button>').join("");
  document.querySelectorAll("#spec .opt").forEach(b=>b.onclick=()=>{c.spec=b.dataset.v; save(); sync();});
  document.querySelectorAll("#primary .chip").forEach(b=>b.onclick=()=>{c.primary=b.dataset.k; save(); sync();});
  const done = ROWS.filter(x=>S[x.review_id].spec!=="" && S[x.review_id].primary!=="").length;
  document.getElementById("prog").textContent = done+" / "+ROWS.length;
}
document.getElementById("note").oninput = e => { S[ROWS[i].review_id].note = e.target.value; save(); };
document.getElementById("prev").onclick = ()=>{ if(i>0){i--; sync();} };
document.getElementById("next").onclick = ()=>{ if(i<ROWS.length-1){i++; sync();} };
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="TEXTAREA") return;
  if(["0","1","2"].includes(e.key)){ S[ROWS[i].review_id].spec = e.key; save(); sync(); }
  if(e.key==="ArrowRight"||e.key==="Enter"){ if(i<ROWS.length-1){i++; sync();} }
  if(e.key==="ArrowLeft"){ if(i>0){i--; sync();} }
});
function q(v){ return '"'+String(v==null?"":v).replace(/"/g,'""')+'"'; }
document.getElementById("export").onclick = ()=>{
  const head = ["review_id","app_name","content","spec","primary","note"].map(q).join(",");
  const body = ROWS.map(r=>{ const c=S[r.review_id];
    return [q(r.review_id),q(r.app_name),q(r.content),q(c.spec),q(c.primary),q(c.note)].join(","); });
  const blob = new Blob(["\\ufeff"+[head].concat(body).join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = __FILE__; a.click();
  const done = ROWS.filter(x=>S[x.review_id].spec!=="" && S[x.review_id].primary!=="").length;
  alert("已导出 " + __FILE__ + "\\n完成 "+done+" / "+ROWS.length+" 条\\n\\n请用导出的文件覆盖 data/ 下的同名文件。");
};
sync();
</script></body></html>"""


def build(sheet: str) -> Path:
    src = DATA / f"coding_{sheet}.csv"
    if not src.exists():
        raise SystemExit(f"缺少 {src}，先运行 scripts/19_make_coding_sheets.py")
    import csv
    with src.open(encoding="utf-8-sig", newline="") as fh:
        rows = [{k: r.get(k, "") for k in ("review_id", "app_name", "content")}
                for r in csv.DictReader(fh)]
    html = (TEMPLATE
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
            .replace("__THEMES__", json.dumps(THEMES, ensure_ascii=False))
            .replace("__THEME_CN__", json.dumps(THEME_CN, ensure_ascii=False))
            .replace("__KEY__", json.dumps(f"coding_{sheet}_v1"))
            .replace("__FILE__", json.dumps(f"coding_{sheet}.csv"))
            .replace("__SHEET__", sheet))
    out = DATA / f"coding_tool_{sheet}.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out}（{len(rows)} 条，{out.stat().st_size // 1024} KB）")
    return out


def main():
    for s in ("A", "B"):
        build(s)


if __name__ == "__main__":
    main()
