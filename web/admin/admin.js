const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function human(n) {
  if (n >= 1048576) return (n / 1048576).toFixed(1) + " MB";
  return (n / 1024).toFixed(1) + " KB";
}
function when(v) {
  if (!v) return "Never";
  const d = typeof v === "number" ? new Date(v * 1000) : new Date(v);
  return isNaN(d) ? String(v) : d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}
async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* no body */ }
  return { ok: res.ok, status: res.status, body };
}
const detail = (r, fallback) => (r.body && (r.body.detail || r.body.message)) || fallback;

/* While a rebuild runs, upload/delete/rebuild are locked (the server answers 409). */
let rebuilding = false;
function syncLocks() {
  $("rebuild").disabled = rebuilding;
  $("file").disabled = rebuilding;
  $("drop").classList.toggle("locked", rebuilding);
  for (const b of document.querySelectorAll("td .btn")) b.disabled = rebuilding;
}

/* ---------- status + table ---------- */
async function refresh() {
  const r = await api("/api/status");
  if (!r.ok) return;
  const s = r.body;
  $("s-docs").textContent = s.exists ? s.num_documents : s.documents.length;
  $("s-chunks").textContent = s.num_chunks;
  $("s-built").textContent = s.exists ? when(s.built_at) : "Not built yet";
  $("s-model").textContent = s.embed_model || "-";
  $("stale").hidden = !(s.stale);
  $("stale").textContent = s.exists ? "Index out of date" : "No index yet";
  $("fresh").hidden = s.stale;
  const body = $("docs");
  body.replaceChildren();
  $("docs-empty").hidden = s.documents.length > 0;
  for (const d of s.documents) {
    const tr = document.createElement("tr");
    tr.append(el("td", "name", d.name));
    const t = el("td"); t.append(el("span", "type", d.type)); tr.append(t);
    tr.append(el("td", "num", d.pages || "-"), el("td", "num", human(d.size)), el("td", "num", d.chunks));
    tr.append(el("td", d.in_index ? "yes" : "no", d.in_index ? "Yes" : "Not yet"));
    const act = el("td", "num");
    const del = el("button", "btn btn-danger", "Delete");
    del.type = "button";
    del.setAttribute("aria-label", "Delete " + d.name);
    del.addEventListener("click", () => confirmDelete(d.name));
    act.append(del); tr.append(act);
    body.append(tr);
  }
  syncLocks();
}

/* ---------- upload ---------- */
const drop = $("drop");
async function upload(fileList) {
  const files = [...fileList];
  if (!files.length || rebuilding) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const list = $("uploads");
  $("rebuild-result").hidden = true;
  list.replaceChildren(el("li", "", "Uploading…"));
  const r = await api("/api/upload", { method: "POST", body: fd });
  list.replaceChildren();
  if (!r.ok || !r.body) {
    const li = el("li", "bad"); li.append(el("span", "tag", "Failed"), el("span", "", "Upload failed. Try again."));
    list.append(li);
  } else {
    for (const s of r.body.saved) {
      const li = el("li", "ok");
      li.append(el("span", "tag", "Added"), el("b", "", s.name), el("span", "", s.replaced ? "Replaced the existing file." : ""));
      list.append(li);
    }
    for (const e of r.body.errors) {
      const li = el("li", "bad");
      li.append(el("span", "tag", "Rejected"), el("b", "", e.name), el("span", "", e.message));
      list.append(li);
    }
  }
  $("file").value = "";
  refresh();
}
$("file").addEventListener("change", (e) => upload(e.target.files));
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));

/* ---------- delete with confirm ---------- */
let pendingDelete = null;
function confirmDelete(name) {
  pendingDelete = name;
  $("dlg-body").textContent = `${name} will be removed from the library. Rebuild the index afterwards to drop it from answers.`;
  $("scrim").hidden = false;
  $("dlg-cancel").focus();
}
function closeDialog() { $("scrim").hidden = true; pendingDelete = null; }
$("dlg-cancel").addEventListener("click", closeDialog);
$("scrim").addEventListener("click", (e) => { if (e.target === $("scrim")) closeDialog(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("scrim").hidden) closeDialog(); });
$("dlg-ok").addEventListener("click", async () => {
  const name = pendingDelete;
  closeDialog();
  if (!name) return;
  $("rebuild-result").hidden = true;
  const r = await api("/api/documents/" + encodeURIComponent(name), { method: "DELETE" });
  const li = el("li", r.ok ? "ok" : "bad");
  li.append(el("span", "tag", r.ok ? "Deleted" : "Failed"), el("b", "", name), el("span", "", r.ok ? "" : detail(r, "Could not delete.")));
  $("uploads").replaceChildren(li);
  refresh();
});

/* ---------- rebuild ---------- */
$("rebuild").addEventListener("click", async () => {
  const btn = $("rebuild"), out = $("rebuild-result");
  rebuilding = true; syncLocks();
  const old = btn.firstElementChild;
  const spin = el("span", "spinner"); btn.replaceChild(spin, old);
  $("rebuild-label").textContent = "Rebuilding…";
  out.hidden = true;
  let r;
  try { r = await api("/api/rebuild", { method: "POST" }); } catch { r = { ok: false, body: null }; }
  btn.replaceChild(old, spin);
  $("rebuild-label").textContent = "Rebuild index";
  rebuilding = false; syncLocks();
  out.hidden = false;
  const good = r.ok && r.body && r.body.ok;
  out.className = "result " + (good ? "ok" : "bad");
  out.textContent = detail(r, "Rebuild failed.");
  refresh();
});

/* ---------- retrieval preview ---------- */
$("prev-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("prev-q").value.trim();
  const msg = $("prev-msg"), hits = $("hits"), btn = $("prev-btn");
  hits.replaceChildren();
  if (!q) { msg.hidden = false; msg.textContent = "Type a question first."; return; }
  btn.disabled = true; btn.textContent = "Searching…";
  const r = await api("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
  btn.disabled = false; btn.textContent = "Preview retrieval";
  const results = (r.body && r.body.results) || [];
  msg.hidden = results.length > 0;
  msg.textContent = (r.body && r.body.message) || (r.ok ? "" : "Preview failed.");
  for (const h of results) {
    const li = el("li", "hit");
    const head = el("div", "hit-head");
    const file = el("div", "hit-file", h.source); file.append(el("small", "", "page " + h.page));
    head.append(el("span", "hit-n", String(h.n)), file, el("span", "hit-score", h.score.toFixed(3)));
    const bar = el("div", "rel-bar"); const fill = document.createElement("i");
    fill.style.width = Math.max(4, Math.min(100, Math.round(h.score * 100))) + "%"; bar.append(fill);
    li.append(head, bar, el("p", "", h.text));
    hits.append(li);
  }
});

refresh();

/* ---------- recent questions ---------- */
const SVG_NS = "http://www.w3.org/2000/svg";
const THUMB = {
  up: "M7 10v11M15 5.9L14 10h5.8a2 2 0 0 1 1.9 2.6l-2.3 8a2 2 0 0 1-1.9 1.4H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.8a2 2 0 0 0 1.8-1.1L12 2a3.1 3.1 0 0 1 3 3.9z",
  down: "M17 14V3M9 18.1l1-4.1H4.2a2 2 0 0 1-1.9-2.6l2.3-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.8a2 2 0 0 0-1.8 1.1L12 22a3.1 3.1 0 0 1-3-3.9z",
};
function thumb(kind) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon");
  const p = document.createElementNS(SVG_NS, "path");
  p.setAttribute("d", THUMB[kind]);
  svg.append(p);
  return svg;
}
const KINDS = {
  answer: ["Answered", "k-ok"], refusal: ["Refused", "k-warn"],
  smalltalk: ["Small talk", "k-muted"], error: ["Error", "k-bad"],
};
function shortTime(v) {
  if (v === null || v === undefined || v === "") return "-";
  const d = typeof v === "number" ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
  if (isNaN(d)) return String(v);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
function latency(ms) {
  if (typeof ms !== "number") return "-";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
}
function renderStats(s) {
  const total = s.total || 0;
  const docQs = (s.answered || 0) + (s.refused || 0) + (s.errors || 0);
  const rate = docQs ? Math.round(((s.answered || 0) / docQs) * 100) + "%" : "";
  const cells = [
    ["Total", total], ["Answered", s.answered || 0, rate], ["Refused", s.refused || 0],
    ["Small talk", s.smalltalk || 0], ["Errors", s.errors || 0], ["Thumbs up", s.up || 0], ["Thumbs down", s.down || 0],
  ];
  const box = $("qstats");
  box.replaceChildren();
  for (const [label, value, extra] of cells) {
    const c = el("div", "qstat");
    const v = el("dd", "", String(value));
    if (extra) v.append(el("small", "", extra));
    c.append(el("dt", "", label), v);
    if (label === "Answered") {
      c.title = `${s.answered || 0} answered of ${docQs} document questions (small talk not counted)`;
      c.append(el("span", "qstat-hint", "of document questions"));
    }
    box.append(c);
  }
}
async function loadQuestions() {
  const btn = $("qs-refresh");
  btn.disabled = true;
  let r;
  try { r = await api("/api/questions?limit=50"); } catch { r = { ok: false, body: null }; }
  btn.disabled = false;
  const body = $("qs");
  body.replaceChildren();
  if (!r.ok || !r.body) {
    renderStats({});
    $("qs-empty").hidden = false;
    $("qs-empty").textContent = "Could not load questions. Try Refresh.";
    return;
  }
  renderStats(r.body.stats || {});
  const items = r.body.items || [];
  $("qs-empty").hidden = items.length > 0;
  $("qs-empty").textContent = "No questions yet. They appear here as people use the chat.";
  for (const it of items) {
    const tr = document.createElement("tr");
    tr.append(el("td", "q-time", shortTime(it.ts)));
    const q = el("td", "q-text");
    const qt = el("div", "clamp", it.question || "");
    qt.title = it.question || "";
    q.append(qt);
    if (it.rewritten) {
      const rw = el("div", "q-rw clamp", "Searched for: " + it.rewritten);
      rw.title = it.rewritten;
      q.append(rw);
    }
    tr.append(q);
    const [label, cls] = KINDS[it.kind] || [String(it.kind || "-"), "k-muted"];
    const k = el("td"); k.append(el("span", "kind " + cls, label)); tr.append(k);
    tr.append(el("td", "num q-lat", latency(it.latency_ms)));
    const c = el("td", "q-cites");
    const cites = it.citations || [];
    if (cites.length) for (const x of cites) c.append(el("span", "cite-tag", x));
    else c.append(el("span", "muted", "-"));
    tr.append(c);
    const f = el("td", "center");
    if (it.feedback === "up" || it.feedback === "down") {
      const s = el("span", "fb fb-" + it.feedback);
      s.setAttribute("aria-label", it.feedback === "up" ? "Thumbs up" : "Thumbs down");
      s.title = s.getAttribute("aria-label");
      s.append(thumb(it.feedback));
      f.append(s);
    } else f.append(el("span", "muted", "-"));
    tr.append(f);
    body.append(tr);
  }
}
$("qs-refresh").addEventListener("click", loadQuestions);
loadQuestions();
