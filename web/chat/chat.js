const FALLBACK_EXAMPLES = [
  "What are the four core functions of the AI RMF?",
  "What are the five Functions of the NIST Privacy Framework?",
  "What fines does the NIST Privacy Framework prescribe for GDPR violations?",
];
const CITE_RE = /\[([^\[\]]+?),\s*p\.\s*(\d+)\]/g;
const REFUSAL_TEXT = "I couldn’t find this in the documents.";
const NET_ERROR = "Could not reach the server. Check that the app is running and try again.";
const STORE_KEY = "docqa.chat.v1";
const HISTORY_TURNS = 6;
const DETAIL_KEY = "docqa.detail";
const CITE_ONLY_RE = /^\s*(?:\[[^\[\]]+?,\s*p\.\s*\d+\]\s*[.,;]?\s*)+$/;

const $ = (id) => document.getElementById(id);
const els = {
  app: $("app"), pill: $("pill"), scope: $("scope"), scroll: $("scroll"), messages: $("messages"),
  examples: $("examples"), form: $("form"), q: $("q"), send: $("send"), count: $("count"),
  newChat: $("new-chat"), toBottom: $("to-bottom"),
};
let busy = false;
let ctrl = null;          // AbortController of the request in flight
let stick = true;
let uid = 0;
let maxChars = 2000;
/* Conversation: {role:"user", content} and {role:"bot", question, history, status, data, feedback, partial, followups}.
   status: pending | streaming | done | stopped. */
let conv = [];
const nodes = new WeakMap(); // record -> <li>

const SVG_NS = "http://www.w3.org/2000/svg";
const ICONS = {
  doc: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h4",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01",
  warn: "M12 8v5M12 17h.01M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  copy: "M9 9h10v11H9zM5 15V4h10",
  check: "M5 12.5l4.5 4.5L19 7",
  arrow: "M7 17L17 7M8 7h9v9",
  ext: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  send: "M12 19V5M5 12l7-7 7 7",
  stop: "M7 7h10v10H7z",
  regen: "M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8M3 3v5h5",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4",
  up: "M7 10v11M15 5.9L14 10h5.8a2 2 0 0 1 1.9 2.6l-2.3 8a2 2 0 0 1-1.9 1.4H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.8a2 2 0 0 0 1.8-1.1L12 2a3.1 3.1 0 0 1 3 3.9z",
  down: "M17 14V3M9 18.1l1-4.1H4.2a2 2 0 0 1-1.9-2.6l2.3-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.8a2 2 0 0 0-1.8 1.1L12 22a3.1 3.1 0 0 1-3-3.9z",
};
function icon(name, extra) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon" + (extra ? " " + extra : ""));
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(SVG_NS, "path");
  p.setAttribute("d", ICONS[name]);
  svg.append(p);
  return svg;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function button(cls, label, ...kids) {
  const b = el("button", cls);
  b.type = "button";
  if (label) { b.setAttribute("aria-label", label); b.title = label; }
  b.append(...kids);
  return b;
}

/* ---------- library info, examples, limit ---------- */
function renderExamples(list) {
  els.examples.replaceChildren();
  for (const q of list.slice(0, 3)) {
    const b = button("card-q", "", el("span", "", q), icon("arrow"));
    b.addEventListener("click", () => ask(q));
    els.examples.append(b);
  }
}
async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    if (Number(info.max_question_chars) > 0) maxChars = Number(info.max_question_chars);
    if (Array.isArray(info.examples) && info.examples.length) renderExamples(info.examples);
    if (info.exists && info.num_documents) {
      const n = info.num_documents;
      els.pill.textContent = `${n} document${n === 1 ? "" : "s"}`;
      els.pill.title = `${info.num_chunks} passages`;
      els.pill.hidden = false;
      const names = info.documents || [];
      const shown = names.slice(0, 2).join(", ");
      els.scope.textContent = names.length > 2 ? `Indexed: ${shown} and ${names.length - 2} more` : `Indexed: ${shown}`;
    } else {
      els.scope.textContent = "No documents indexed yet. Ask the owner to add documents.";
    }
  } catch {
    els.scope.textContent = "Library status unavailable";
  }
  syncSend();
}

/* ---------- scrolling ---------- */
const nearBottom = () => els.scroll.scrollHeight - els.scroll.scrollTop - els.scroll.clientHeight < 80;
function toBottom(smooth = true) {
  els.scroll.scrollTo({ top: els.scroll.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}
els.scroll.addEventListener("scroll", () => {
  stick = nearBottom();
  els.toBottom.hidden = stick || els.app.classList.contains("is-empty");
});
els.toBottom.addEventListener("click", () => { stick = true; toBottom(); });

/* ---------- sources ---------- */
function findSource(sources, m) {
  return sources.find((x) => x.source === m[1].trim() && String(x.page) === m[2]);
}
function numberSources(text, sources) {
  const order = [];
  for (const m of text.matchAll(CITE_RE)) {
    const s = findSource(sources, m);
    if (s && !order.includes(s)) order.push(s);
  }
  for (const s of sources) if (!order.includes(s)) order.push(s);
  return order;
}
/* One entry per file + page: a page cited through several chunks gets one number. */
function groupSources(sources) {
  const groups = new Map();
  for (const s of sources || []) {
    const key = s.source + "|" + s.page;
    const g = groups.get(key);
    if (g) { g.text += "\n\n…\n\n" + s.text; g.score = Math.max(g.score, s.score); }
    else groups.set(key, { ...s });
  }
  return [...groups.values()];
}

/* ---------- markdown subset, DOM nodes only ---------- */
const INLINE_RE = /`([^`\n]+)`|\*\*(.+?)\*\*|(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])|\[([^\[\]]+?),\s*p\.\s*(\d+)\]/g;
/* Put a citation marker in a nowrap span with the word before it, so it never wraps onto a line alone. */
function glue(parent, before) {
  const lead = before.replace(/\s+$/, "");
  const m = /\S+$/.exec(lead);
  let word = "";
  if (m) {
    parent.append(document.createTextNode(lead.slice(0, m.index)));
    word = m[0];
  } else {
    const prev = parent.lastChild;
    if (prev && prev.nodeType === 1 && prev.classList.contains("glue")) return prev; // [a][b]
    if (prev && prev.nodeType === 3) {
      const t = prev.data.replace(/\s+$/, "");
      const k = /\S+$/.exec(t);
      if (k) { prev.data = t.slice(0, k.index); word = k[0]; }
    }
    if (!word) { parent.append(document.createTextNode(before)); return parent; }
  }
  const g = el("span", "glue", word + " ");
  parent.append(g);
  return g;
}
function inline(text, parent, ctx) {
  let last = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    const before = text.slice(last, m.index);
    if (m[5] === undefined) parent.append(document.createTextNode(before));
    last = m.index + m[0].length;
    if (m[1] !== undefined) parent.append(el("code", "", m[1]));
    else if (m[2] !== undefined) { const b = el("strong"); inline(m[2], b, ctx); parent.append(b); }
    else if (m[3] !== undefined) { const i = el("em"); inline(m[3], i, ctx); parent.append(i); }
    else {
      const s = ctx.resolve ? ctx.resolve(m[4].trim(), m[5]) : findSource(ctx.sources, [m[0], m[4], m[5]]);
      if (!s) { parent.append(document.createTextNode(before)); continue; } // tag for a passage we don't have: drop it
      const n = ctx.numOf(s);
      const host = glue(parent, before);
      if (ctx.resolve) { host.append(el("span", "cite-n", String(n))); continue; } // still streaming
      const b = button("cite-n", `Source ${n}: ${s.source}, page ${s.page}`, String(n));
      b.removeAttribute("title");
      b.dataset.n = n;
      ctx.markers.push(b);
      host.append(b);
    }
  }
  parent.append(document.createTextNode(text.slice(last)));
}
function renderMarkdown(text, ctx) {
  const root = el("div", "answer");
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  let para = null, list = null;
  const flush = () => { para = null; list = null; };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flush(); continue; }
    const h = /^\s{0,3}#{1,6}\s+(.*)$/.exec(line);
    const ul = /^\s{0,3}[-*•]\s+(.*)$/.exec(line);
    const ol = /^\s{0,3}(\d{1,3})[.)]\s+(.*)$/.exec(line);
    if (h) {
      flush();
      const p = el("p", "md-h"); const s = el("strong"); inline(h[1].replace(/\s#+$/, ""), s, ctx); p.append(s);
      root.append(p);
    } else if (ul || ol) {
      const tag = ul ? "UL" : "OL";
      if (!list || list.tagName !== tag) {
        para = null;
        list = el(tag.toLowerCase());
        if (ol && ol[1] !== "1") list.start = Number(ol[1]);
        root.append(list);
      }
      const li = el("li"); inline(ul ? ul[1] : ol[2], li, ctx); list.append(li);
    } else if (list && /^\s+/.test(raw) && list.lastElementChild) {
      // indented continuation of a list item
      const li = list.lastElementChild; li.append(document.createTextNode(" ")); inline(line.trim(), li, ctx);
    } else if (CITE_ONLY_RE.test(line) && root.lastElementChild) {
      // a "paragraph" of only citation tags belongs to the block before it (e.g. the last list item)
      inline(" " + line.trim(), lastBlock(root), ctx);
    } else {
      list = null;
      if (para) para.append(el("br"));
      else { para = el("p"); root.append(para); }
      inline(line, para, ctx);
    }
  }
  return root;
}

function lastBlock(root) {
  const b = root.lastElementChild;
  return b && (b.tagName === "UL" || b.tagName === "OL") && b.lastElementChild ? b.lastElementChild : b;
}

/* ---------- streaming view ---------- */
/* Hide what can't render yet: an unclosed trailing [tag, an odd ** or ` on the last line. */
function cleanPartial(text) {
  let t = String(text || "").replace(/\[[^\]\n]*$/, "");
  const tail = t.slice(t.lastIndexOf("\n") + 1);
  for (const mark of ["**", "`"]) {
    if (tail.split(mark).length % 2 === 0) { const i = t.lastIndexOf(mark); t = t.slice(0, i) + t.slice(i + mark.length); }
  }
  return t;
}
function streamCtx() {
  const seen = [];
  return {
    sources: [], markers: [],
    resolve(source, page) {
      let s = seen.find((x) => x.source === source && String(x.page) === page);
      if (!s) { s = { source, page }; seen.push(s); }
      return s;
    },
    numOf: (s) => seen.indexOf(s) + 1,
  };
}
function streamView(text, live) {
  const root = renderMarkdown(cleanPartial(text), streamCtx());
  root.classList.add(live ? "streaming" : "partial");
  if (live) {
    const c = el("span", "caret");
    c.setAttribute("aria-hidden", "true");
    (lastBlock(root) || root).append(c);
  }
  return root;
}
function paintStream(rec) {
  rec.raf = 0;
  if (rec.status !== "streaming") return;
  const li = nodes.get(rec);
  const box = li && li.querySelector(".stream");
  if (box) box.replaceChildren(streamView(rec.partial, true));
  else renderBot(rec);
  if (stick) toBottom(false);
}
function queuePaint(rec) { if (!rec.raf) rec.raf = requestAnimationFrame(() => paintStream(rec)); }

function buildAnswer(raw, withSources) {
  const sources = withSources ? groupSources(raw.sources) : [];
  const order = numberSources(raw.answer || "", sources);
  const ctx = { sources, markers: [], numOf: (s) => order.indexOf(s) + 1 };
  const wrap = el("div");
  wrap.append(renderMarkdown(raw.answer, ctx));
  const markers = ctx.markers;

  if (order.length) {
    const id = "a" + ++uid;
    const row = el("div", "srcs");
    row.append(el("span", "srcs-label", "Sources"));
    const chips = [], panels = [];
    const holder = el("div");
    order.forEach((s, i) => {
      const n = i + 1;
      const chip = el("button", "src-chip");
      chip.type = "button";
      chip.id = `${id}-c${n}`;
      chip.setAttribute("aria-expanded", "false");
      chip.setAttribute("aria-controls", `${id}-p${n}`);
      chip.append(el("span", "num", String(n)), el("span", "nm", s.source), el("span", "pg", `p.${s.page}`));
      chips.push(chip);
      row.append(chip);

      const panel = el("div", "panel");
      panel.id = `${id}-p${n}`;
      panel.setAttribute("role", "region");
      panel.setAttribute("aria-labelledby", chip.id);
      const inner = el("div", "panel-in");
      const card = el("div", "panel-card");
      const meta = el("div", "panel-meta");
      const rel = el("div", "rel");
      const bar = el("span", "rel-bar"), fill = document.createElement("i");
      const score = Number(s.score) || 0;
      fill.style.width = Math.max(4, Math.min(100, Math.round(score * 100))) + "%";
      bar.append(fill);
      rel.append(el("span", "", "relevance"), bar, el("span", "", score.toFixed(2)));
      meta.append(el("b", "", s.source), el("span", "pg", `page ${s.page}`), rel);
      const open = el("a", "open-pg");
      open.href = `/docs/${encodeURIComponent(s.source)}#page=${encodeURIComponent(s.page)}`;
      open.target = "_blank";
      open.rel = "noopener noreferrer";
      open.setAttribute("aria-label", `Open ${s.source} at page ${s.page} in a new tab`);
      open.append(el("span", "", `Open page ${s.page}`), icon("ext"));
      const foot = el("div", "panel-foot");
      foot.append(open);
      card.append(meta, el("p", "passage", s.text), foot);
      inner.append(card);
      panel.append(inner);
      panels.push(panel);
      holder.append(panel);
    });

    let open = -1;
    const setOpen = (i) => {
      open = i;
      chips.forEach((c, k) => c.setAttribute("aria-expanded", String(k === i)));
      panels.forEach((p, k) => { p.classList.toggle("open", k === i); });
      markers.forEach((m) => m.classList.toggle("on", Number(m.dataset.n) === i + 1));
    };
    chips.forEach((c, i) => c.addEventListener("click", () => setOpen(open === i ? -1 : i)));
    markers.forEach((m) => m.addEventListener("click", () => {
      const i = Number(m.dataset.n) - 1;
      setOpen(open === i ? -1 : i);
      if (open === i) setTimeout(() => panels[i].scrollIntoView({ behavior: "smooth", block: "nearest" }), 120);
    }));
    wrap.append(row, holder);
  }
  return wrap;
}

/* ---------- message actions ---------- */
function copyButton(getText) {
  const b = button("act", "Copy answer", icon("copy"));
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(getText()); } catch { return; }
    b.replaceChildren(icon("check"), el("span", "", "Copied"));
    setTimeout(() => b.replaceChildren(icon("copy")), 1600);
  });
  return b;
}
function regenButton(rec) {
  const b = button("act regen", "Regenerate", icon("regen"));
  b.addEventListener("click", () => { if (!busy) { stick = true; run(rec); } });
  return b;
}
function feedbackButtons(rec) {
  const group = el("span", "fb");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Rate this answer");
  const make = (rating, label) => {
    const b = button("act thumb", label, icon(rating));
    b.dataset.rating = rating;
    b.setAttribute("aria-pressed", String(rec.feedback === rating));
    b.addEventListener("click", () => rate(rec, rating, group));
    return b;
  };
  group.append(make("up", "Good answer"), make("down", "Bad answer"));
  return group;
}
async function rate(rec, rating, group) {
  if (!rec.data || !rec.data.id || rec.feedback === rating) return;
  rec.feedback = rating;
  for (const b of group.children) b.setAttribute("aria-pressed", String(b.dataset.rating === rating));
  save();
  try {
    const res = await fetch("/api/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: rec.data.id, rating }),
    });
    if (!res.ok) console.warn("feedback not saved:", res.status);
  } catch (e) {
    console.warn("feedback not saved:", e);
  }
}
function suggestionChips(list) {
  const row = el("div", "sugg");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", "Suggested questions");
  for (const q of list.slice(0, 3)) {
    const b = button("sugg-chip", "", q);
    b.addEventListener("click", () => ask(q));
    row.append(b);
  }
  return row;
}

function relatedRow(list) {
  const wrap = el("div", "related");
  wrap.setAttribute("role", "group");
  wrap.setAttribute("aria-label", "Related questions");
  const row = el("div", "related-list");
  for (const q of list) {
    const b = button("sugg-chip", "", q);
    b.addEventListener("click", () => ask(q));
    row.append(b);
  }
  wrap.append(el("span", "related-label", "Related"), row);
  return wrap;
}
async function loadFollowups(rec, data) {
  let list;
  try {
    const res = await fetch("/api/followups", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: rec.question.slice(0, 2000), answer: String(data.answer).slice(0, 2000) }),
    });
    if (!res.ok) return;
    const j = await res.json();
    list = (Array.isArray(j && j.followups) ? j.followups : [])
      .filter((q) => typeof q === "string" && q.trim()).map((q) => q.trim()).slice(0, 3);
  } catch { return; }
  if (rec.data !== data || rec.status !== "done" || !list.length) return; // regenerated, reset, or nothing to show
  rec.followups = list;
  save();
  const li = nodes.get(rec);
  const body = li && li.querySelector(".body");
  if (!body || body.querySelector(".related")) return;
  const row = relatedRow(list);
  row.classList.add("enter");
  body.insertBefore(row, body.querySelector(":scope > .actions"));
  if (stick) toBottom();
}

/* ---------- rendering ---------- */
function rewriteLine(text) {
  const r = el("p", "rewrite");
  r.append(icon("search"), el("span", "", "Searched for:"), el("span", "rw-q", text));
  return r;
}
function kindOf(d) {
  if (!d) return "error";
  if (d.kind) return d.kind;
  return d.error ? "error" : d.refused ? "refusal" : "answer";
}
function addUserNode(rec) {
  const li = el("li", "msg msg-user");
  li.append(el("div", "bubble", rec.content));
  els.messages.append(li);
  nodes.set(rec, li);
}
function addBotNode(rec) {
  const li = el("li", "msg msg-bot");
  const mark = el("div", "mark"); mark.append(icon("doc"));
  li.append(mark, el("div", "body"));
  els.messages.append(li);
  nodes.set(rec, li);
  renderBot(rec);
}
function retryButton(rec, cls) {
  const b = button(cls, "", "Retry");
  b.addEventListener("click", () => { if (!busy) { stick = true; run(rec); } });
  return b;
}
function renderBot(rec) {
  const li = nodes.get(rec);
  if (!li) return;
  const mark = li.querySelector(".mark");
  const body = li.querySelector(".body");
  const live = rec.status === "pending" || rec.status === "streaming";
  mark.classList.toggle("busy", live);
  const rw = rec.meta && rec.meta.rewritten;
  if (live) {
    const parts = rw ? [rewriteLine(rw)] : [];
    if (rec.partial) {
      const box = el("div", "stream");
      box.append(streamView(rec.partial, true));
      parts.push(box);
    } else {
      const t = el("div", "typing");
      t.append(el("span", "shimmer", rec.meta ? "Writing the answer…" : "Searching the documents…"));
      parts.push(t);
    }
    body.replaceChildren(...parts);
    return;
  }
  if (rec.status === "stopped") {
    const parts = rw ? [rewriteLine(rw)] : [];
    if (rec.partial) parts.push(streamView(rec.partial, false));
    const s = el("div", "stopped");
    s.append(el("span", "", "Stopped."), retryButton(rec, "btn btn-sm"));
    parts.push(s);
    body.replaceChildren(...parts);
    return;
  }
  const d = rec.data || {};
  const kind = kindOf(d);
  const parts = [];
  if (d.rewritten && kind !== "error") parts.push(rewriteLine(d.rewritten));
  if (kind === "error") {
    const e = el("div", "err");
    e.setAttribute("role", "alert");
    e.append(icon("warn"), el("span", "", d.error || d.answer || "Something went wrong."), retryButton(rec, "btn"));
    parts.push(e);
  } else if (kind === "refusal") {
    const n = el("div", "note");
    n.append(icon("info"), el("span", "", REFUSAL_TEXT));
    parts.push(n);
  } else {
    parts.push(buildAnswer(d, kind === "answer"));
  }
  if (Array.isArray(d.suggestions) && d.suggestions.length) parts.push(suggestionChips(d.suggestions));
  if (kind === "answer" && Array.isArray(rec.followups) && rec.followups.length) parts.push(relatedRow(rec.followups));
  if (kind === "answer" || kind === "refusal") {
    const actions = el("div", "actions");
    if (kind === "answer") actions.append(copyButton(() => d.answer || ""));
    actions.append(regenButton(rec));
    if (d.id) actions.append(el("span", "act-sep"), feedbackButtons(rec));
    parts.push(actions);
  }
  body.replaceChildren(...parts);
}

/* ---------- persistence ---------- */
function save() {
  try {
    const out = conv
      .filter((r) => r.role === "user" || r.status !== "pending")
      .map((r) => r.role === "user" ? { role: "user", content: r.content }
        : { role: "bot", question: r.question, history: r.history, status: r.status, data: r.data, feedback: r.feedback || null,
            partial: r.status === "stopped" ? r.partial || "" : "", followups: r.followups || null, meta: r.status === "stopped" ? r.meta || null : null });
    if (out.length) localStorage.setItem(STORE_KEY, JSON.stringify(out));
    else localStorage.removeItem(STORE_KEY);
  } catch { /* storage unavailable: the conversation just isn't kept */ }
}
function restore() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null"); } catch { saved = null; }
  if (!Array.isArray(saved) || !saved.length) return;
  for (const r of saved) {
    if (!r || (r.role !== "user" && r.role !== "bot")) continue;
    if (r.role === "user") {
      const rec = { role: "user", content: String(r.content || "") };
      conv.push(rec); addUserNode(rec);
    } else {
      const rec = { role: "bot", question: String(r.question || ""), history: Array.isArray(r.history) ? r.history : [],
        status: r.status === "stopped" ? "stopped" : "done", data: r.data || null, feedback: r.feedback || null,
        partial: typeof r.partial === "string" ? r.partial : "",
        meta: r.meta && typeof r.meta === "object" ? { id: r.meta.id || null, rewritten: r.meta.rewritten || null } : null,
        followups: Array.isArray(r.followups) ? r.followups.filter((q) => typeof q === "string").slice(0, 3) : null };
      conv.push(rec); addBotNode(rec);
    }
  }
  if (conv.length) {
    els.app.classList.remove("is-empty");
    requestAnimationFrame(() => toBottom(false));
  }
}

/* ---------- asking ---------- */
function buildHistory() {
  const msgs = [];
  for (const r of conv) {
    if (r.role === "user") msgs.push({ role: "user", content: r.content });
    else if (r.status === "done" && r.data && kindOf(r.data) !== "error" && r.data.answer) {
      msgs.push({ role: "assistant", content: r.data.answer });
    }
  }
  return msgs.slice(-HISTORY_TURNS);
}

function detail() {
  const c = document.querySelector('input[name="detail"]:checked');
  return c && c.value === "brief" ? "brief" : "detailed";
}

/* Read `data: {...}` events (blank-line separated, possibly split across chunks). Returns the done payload. */
async function readStream(res, rec, mine) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", final = null;
  const handle = (block) => {
    const json = block.split(/\r?\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
    if (!json) return;
    let ev;
    try { ev = JSON.parse(json); } catch { return; }
    if (ctrl !== mine || !ev) return;
    if (ev.type === "meta") {
      rec.meta = { id: ev.id || null, rewritten: ev.rewritten || null };
      rec.status = "streaming";
      if (!rec.raf) renderBot(rec);
    } else if (ev.type === "delta" && typeof ev.text === "string") {
      rec.partial += ev.text;
      rec.status = "streaming";
      queuePaint(rec);
    } else if (ev.type === "done") {
      final = ev;
    }
  };
  const drain = () => {
    let m;
    while ((m = /\r?\n\r?\n/.exec(buf))) { handle(buf.slice(0, m.index)); buf = buf.slice(m.index + m[0].length); }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    drain();
  }
  buf += dec.decode();
  drain();
  if (buf.trim()) handle(buf);
  if (!final) return null;
  const { type, ...data } = final;
  return data;
}

async function run(rec) {
  busy = true;
  ctrl = new AbortController();
  const mine = ctrl;
  rec.status = "pending"; rec.feedback = null; rec.partial = ""; rec.meta = null; rec.followups = null;
  renderBot(rec); syncSend();
  if (stick) toBottom();
  const payload = JSON.stringify({ question: rec.question, history: rec.history, detail: detail() });
  const post = (url) => fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: payload, signal: mine.signal,
  });
  let data = null, status = "done";
  try {
    let res = await post("/api/ask/stream");
    if (res.status === 404 || res.status === 405) { // older backend without streaming
      res = await post("/api/ask");
      if (!res.ok) throw new Error(String(res.status));
      data = await res.json();
    } else {
      if (!res.ok || !res.body) throw new Error(String(res.status));
      data = await readStream(res, rec, mine);
      if (!data) throw new Error("stream ended without a result");
    }
  } catch (e) {
    if (e && e.name === "AbortError") status = "stopped";
    else data = { kind: "error", answer: "", refused: false, sources: [], suggestions: [], error: NET_ERROR };
  }
  if (ctrl !== mine) return; // conversation was reset meanwhile
  if (rec.raf) { cancelAnimationFrame(rec.raf); rec.raf = 0; }
  rec.status = status; rec.data = data;
  if (status === "done") { rec.partial = ""; rec.meta = null; }
  ctrl = null; busy = false;
  renderBot(rec); syncSend(); save();
  if (stick) toBottom();
  if (status === "done" && kindOf(data) === "answer" && data.answer) loadFollowups(rec, data);
}

function ask(question) {
  question = String(question).trim();
  if (!question || busy || question.length > maxChars) return;
  els.app.classList.remove("is-empty");
  const history = buildHistory();
  const user = { role: "user", content: question };
  const bot = { role: "bot", question, history, status: "pending", data: null, feedback: null };
  conv.push(user, bot);
  addUserNode(user); addBotNode(bot);
  els.q.value = ""; autosize();
  stick = true;
  run(bot);
}

/* ---------- composer ---------- */
function setSendMode(stop) {
  if (els.send.classList.contains("stop") === stop && els.send.firstElementChild) return;
  els.send.classList.toggle("stop", stop);
  els.send.replaceChildren(icon(stop ? "stop" : "send", stop ? "solid" : ""));
  els.send.setAttribute("aria-label", stop ? "Stop answering" : "Send question");
}
function syncSend() {
  const n = els.q.value.length;
  const over = n > maxChars;
  els.count.hidden = n < Math.floor(maxChars * 0.9);
  els.count.textContent = `${n} / ${maxChars}`;
  els.count.classList.toggle("over", over);
  els.form.classList.toggle("over", over);
  setSendMode(busy);
  els.send.disabled = busy ? false : (!els.q.value.trim() || over);
}
function autosize() { els.q.style.height = "auto"; els.q.style.height = Math.min(els.q.scrollHeight, 200) + "px"; syncSend(); }

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (busy) { if (ctrl) ctrl.abort(); return; }
  ask(els.q.value);
});
els.q.addEventListener("input", autosize);
els.q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!busy && !els.send.disabled) els.form.requestSubmit();
  }
});
els.newChat.addEventListener("click", () => {
  if (ctrl) ctrl.abort();
  ctrl = null; busy = false;
  conv = [];
  save();
  els.messages.replaceChildren();
  els.app.classList.add("is-empty");
  els.toBottom.hidden = true;
  els.q.value = ""; autosize(); els.q.focus();
});

for (const r of document.querySelectorAll('input[name="detail"]')) {
  r.addEventListener("change", () => { try { localStorage.setItem(DETAIL_KEY, detail()); } catch { /* not kept */ } });
}
try { if (localStorage.getItem(DETAIL_KEY) === "brief") $("detail-brief").checked = true; } catch { /* default: detailed */ }

renderExamples(FALLBACK_EXAMPLES);
restore();
syncSend();
loadInfo();
els.q.focus();
