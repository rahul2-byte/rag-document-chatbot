// Scripted screen recording of the RAG chatbot demo (new chat + owner console UI). See DEMO_SCRIPT.md.
// Run from the repo root, with both apps already running (chat :7860, owner console :7861):
//   npm install --no-save playwright && npx playwright install chromium   (once; ffmpeg/ffprobe must be on PATH)
//   node demo/record_demo.cjs
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFileSync, spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const PDF = 'NIST.CSWP.29.pdf';
const STAGED = path.join(__dirname, 'staging', PDF);
const OUT = path.join(__dirname, 'out');
const RAW = path.join(OUT, 'raw');
const BACKUP = process.env.DEMO_BACKUP_DIR || path.join(os.tmpdir(), 'chatbot_demo_backup');
const CHAT = 'http://127.0.0.1:7860', ADMIN = 'http://127.0.0.1:7861';
const Q1 = 'What are the six Functions of the CSF 2.0?';
const Q_FALLBACK = 'Which of those covers incident recovery?';
const Q_OFF = 'What is the capital of France?';
const ANSWER_TIMEOUT = +(process.env.DEMO_ANSWER_TIMEOUT_MS || 60000);
const MAX_TOTAL = 74; // seconds, hard budget for the video
const FFMPEG = process.env.FFMPEG || 'ffmpeg', FFPROBE = process.env.FFPROBE || 'ffprobe';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sh = (cmd, args) => execFileSync(cmd, args, { stdio: ['ignore', 'pipe', 'inherit'] }).toString();
const probeDur = (f) => +sh(FFPROBE, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f]).trim();

async function adminApi(method, p) {
  const res = await fetch(ADMIN + p, { method, headers: { Origin: ADMIN } });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(`${method} ${p} -> ${res.status} ${JSON.stringify(body)}`);
  return body;
}
const status = () => adminApi('GET', '/api/status');

// data/ fingerprint by name + content (mtime is expected to change after the on-camera upload)
function dataDigest(dir) {
  return fs.readdirSync(dir).sort().map((f) => {
    const h = crypto.createHash('sha1').update(fs.readFileSync(path.join(dir, f))).digest('hex');
    return `${f}:${h}`;
  }).join('\n');
}

function card(body) {
  return `<html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#16191b;font-family:system-ui,sans-serif;color:#f4f4f2;text-align:center">${body}</body></html>`;
}
const TITLE = card(`<div><h1 style="font-size:60px;margin:0 0 18px;font-weight:600">Document Q&amp;A</h1>
<p style="font-size:28px;margin:0;color:#b9bec2">Ask your documents. Every answer cited.</p></div>`);
const END = card(`<div style="font-size:34px;line-height:1.8;color:#e6e8e9">Cited answers &middot; Refuses when unsure &middot; Owner console</div>`);

// Small visible cursor so viewers can follow clicks (headless video has no pointer).
const CURSOR = `(() => {
  if (document.contentType !== 'text/html') return;
  const add = () => {
    if (!document.body || document.getElementById('__demo_cursor')) return;
    const c = document.createElement('div'); c.id = '__demo_cursor';
    c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;background:rgba(20,20,20,.35);border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);pointer-events:none;z-index:2147483647;transition:left .08s linear,top .08s linear';
    document.body.append(c);
    addEventListener('mousemove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', add); else add();
})();`;

async function moveTo(page, loc) {
  await loc.scrollIntoViewIfNeeded();
  const b = await loc.boundingBox();
  if (b) await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
}
async function clickHuman(page, loc, opts) { await moveTo(page, loc); await sleep(150); await loc.click(opts); }

const lastBot = (page) => page.locator('li.msg-bot').last();
async function waitDone(page) {
  // done = the last bot message has rendered its final state (actions row, refusal note or error)
  await lastBot(page).locator('.actions, .note, .err').first().waitFor({ timeout: ANSWER_TIMEOUT });
}
async function typeAndSend(page, q) {
  const box = page.locator('#q');
  await clickHuman(page, box);
  await box.pressSequentially(q, { delay: 25 });
  await sleep(300);
  await box.press('Enter');
  await page.locator('li.msg-user').last().waitFor();
  await page.mouse.move(1180, 330, { steps: 8 }); // park the cursor off the text
}

(async () => {
  const tl = []; // [label, start s, end s]
  let browser, ctx, t0, mark = 0, popupInfo = null, setupDone = false;
  const now = () => (Date.now() - t0) / 1000;
  const step = (label) => { const t = now(); tl.push([label, mark, t]); mark = t; };
  const shots = {};
  const shot = async (pg, name) => { await pg.screenshot({ path: path.join(OUT, name) }); shots[name] = now(); };
  let before;
  try {
    // ---------- setup (not recorded) ----------
    if (!fs.existsSync(STAGED)) {
      // demo/staging/ is gitignored: stage the upload copy from data/ before setup removes it from the live system.
      const src = path.join(ROOT, 'data', PDF);
      if (!fs.existsSync(src)) throw new Error(`missing ${STAGED} and ${src}; put ${PDF} in data/ first`);
      fs.mkdirSync(path.dirname(STAGED), { recursive: true });
      fs.copyFileSync(src, STAGED);
      console.log(`Staged ${PDF} to ${STAGED}`);
    }
    fs.mkdirSync(OUT, { recursive: true });
    const st0 = await status();
    if (!st0.documents.some((d) => d.name === PDF)) throw new Error(`${PDF} is not in data/ before the run; restore it first`);
    fs.rmSync(BACKUP, { recursive: true, force: true });
    fs.mkdirSync(BACKUP, { recursive: true });
    sh('cp', ['-a', path.join(ROOT, 'data'), path.join(ROOT, 'index'), BACKUP]);
    before = dataDigest(path.join(ROOT, 'data'));
    setupDone = true;
    console.log(`Backed up data/ and index/ to ${BACKUP}`);
    await adminApi('DELETE', `/api/documents/${encodeURIComponent(PDF)}`);
    const rb = await adminApi('POST', '/api/rebuild');
    console.log(`Setup: removed ${PDF}, rebuilt index (${rb.num_chunks} chunks, ${rb.seconds}s)`);
    for (const f of fs.readdirSync(OUT)) if (/^frame_.*\.png$/.test(f)) fs.rmSync(path.join(OUT, f));
    fs.rmSync(RAW, { recursive: true, force: true });

    browser = await chromium.launch({ headless: true, channel: 'chromium' }); // full Chromium: has the PDF viewer
    ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, recordVideo: { dir: RAW, size: { width: 1280, height: 720 } } });
    await ctx.addInitScript(CURSOR);
    t0 = Date.now();
    const page = await ctx.newPage();

    // 1) Title card
    await page.setContent(TITLE);
    await sleep(3000);
    step('Title card');

    // 2) Owner console: upload + rebuild
    await page.goto(ADMIN, { waitUntil: 'commit' });
    const tTitleEnd = now(); // dark title -> light console: used to sync wall clock with the video
    await page.waitForLoadState('networkidle');
    await page.locator('#docs tr').nth(1).waitFor();
    await sleep(1200);
    await moveTo(page, page.locator('#drop'));
    await sleep(400);
    await page.locator('#file').setInputFiles(STAGED);
    await page.locator('#uploads li.ok').waitFor({ timeout: 20000 });
    await page.locator('#stale').waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
    await sleep(1000);
    await shot(page, 'frame_upload.png');
    await clickHuman(page, page.locator('#rebuild'));
    await page.locator('#rebuild-result').waitFor({ state: 'visible', timeout: 180000 });
    await page.locator('#fresh').waitFor({ state: 'visible', timeout: 10000 });
    await page.waitForFunction(() => document.getElementById('s-docs').textContent === '3', null, { timeout: 10000 });
    await sleep(2500);
    step('Owner console: upload + rebuild');

    // 3) Chat: streamed, cited answer (Detailed mode)
    await page.goto(CHAT, { waitUntil: 'networkidle' });
    await page.evaluate(() => localStorage.clear());
    await page.reload({ waitUntil: 'networkidle' });
    await page.locator('#pill').waitFor({ state: 'visible' });
    await sleep(600);
    await clickHuman(page, page.locator('label:has(#detail-detailed)'));
    await sleep(300);
    await typeAndSend(page, Q1);
    const tAsk1 = now();
    await page.waitForFunction(() => {
      const s = [...document.querySelectorAll('li.msg-bot')].pop()?.querySelector('.stream');
      return s && s.innerText.length > 220;
    }, null, { timeout: ANSWER_TIMEOUT });
    await shot(page, 'frame_streaming.png');
    await waitDone(page);
    const ans1 = now() - tAsk1;
    await sleep(1500);
    await shot(page, 'frame_cited_answer.png');
    await sleep(1500);
    step(`Chat: ask + streamed answer (${ans1.toFixed(1)}s to done)`);

    // 4) Source passage, then open the PDF page in a new tab
    const chips = lastBot(page).locator('.src-chip');
    const pref = chips.filter({ hasText: 'CSWP.29' });
    const chip = (await pref.count()) ? pref.first() : chips.first();
    await clickHuman(page, chip);
    const panel = lastBot(page).locator('.panel.open');
    await panel.waitFor();
    await sleep(300);
    await panel.locator('.panel-card').scrollIntoViewIfNeeded();
    await sleep(2000);
    const link = panel.locator('a.open-pg');
    await moveTo(page, link);
    await sleep(500);
    const tPopup = { open: 0, close: 0 };
    const popupP = ctx.waitForEvent('page');
    await link.click();
    const popup = await popupP;
    tPopup.open = now();
    await popup.waitForLoadState('load').catch(() => {});
    await sleep(3000);
    await shot(popup, 'frame_source_open.png');
    tPopup.close = now();
    await popup.close();
    popupInfo = { video: await popup.video().path(), open: tPopup.open, close: tPopup.close, url: popup.url() };
    await page.bringToFront();
    step('Source passage + PDF page in new tab');

    // 5) Related follow-up
    const rel = lastBot(page).locator('.related .sugg-chip');
    let followQ;
    try {
      await rel.first().waitFor({ timeout: 8000 });
      followQ = (await rel.first().innerText()).trim();
      await page.evaluate(() => { const s = document.getElementById('scroll'); s.scrollTo({ top: s.scrollHeight, behavior: 'smooth' }); });
      await sleep(900);
      await moveTo(page, rel.first());
      await sleep(1300);
      await clickHuman(page, rel.first());
      await page.mouse.move(1180, 330, { steps: 8 });
    } catch {
      followQ = Q_FALLBACK;
      await typeAndSend(page, Q_FALLBACK);
    }
    await waitDone(page);
    await sleep(2000);
    await shot(page, 'frame_followup.png');
    await sleep(1000);
    step(`Chat: related follow-up ("${followQ}")`);

    // 6) Off-topic -> refusal
    await typeAndSend(page, Q_OFF);
    await waitDone(page);
    await sleep(2500);
    await shot(page, 'frame_refusal.png');
    step('Chat: off-topic question refused');

    // 7) Owner console: recent questions
    await page.goto(ADMIN, { waitUntil: 'networkidle' });
    await page.locator('#qs tr').first().waitFor({ timeout: 10000 });
    await sleep(500);
    await page.locator('#h-qs').evaluate((h) => h.closest('section').scrollIntoView({ behavior: 'smooth', block: 'start' }));
    await sleep(Math.max(2500, Math.min(4000, (MAX_TOTAL - 5 - now()) * 1000)));
    await shot(page, 'frame_owner_panel.png');
    step('Owner console: recent questions');

    // 8) End card
    await page.setContent(END);
    await sleep(3000);
    step('End card');
    const tEnd = now();

    await ctx.close(); ctx = null;
    const mainVid = await page.video().path();
    const mp4 = path.join(OUT, 'demo.mp4');
    // Sync wall clock -> video time at the title-card cut (first big scene change after 1 s).
    const sc = spawnSync(FFMPEG, ['-hide_banner', '-i', mainVid, '-vf', "select='gt(scene,0.4)',showinfo", '-f', 'null', '-']).stderr.toString();
    const cuts = [...sc.matchAll(/pts_time:([\d.]+)/g)].map((m) => +m[1]);
    const cut = cuts.find((t) => t > 1);
    const off = cut === undefined ? 0 : cut - tTitleEnd;
    console.log(`Video sync: title cut at ${cut}s video / ${tTitleEnd.toFixed(2)}s wall -> offset ${off.toFixed(2)}s`);
    const A = Math.max(0, popupInfo.open + off), C = Math.max(A, popupInfo.close + off), D = C - A;
    // The new tab is a separate recorded page: splice it into the timeline for the real time it was open.
    const enc = ['-t', (tEnd + off).toFixed(3), '-r', '30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4];
    const fc = `[0:v]fps=30,format=yuv420p,split=2[m1][m2];[m1]trim=0:${A.toFixed(3)},setpts=PTS-STARTPTS[a];` +
      `[1:v]fps=30,format=yuv420p,scale=1280:720,tpad=stop_mode=clone:stop_duration=2,trim=0:${D.toFixed(3)},setpts=PTS-STARTPTS[b];` +
      `[m2]trim=start=${C.toFixed(3)},setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[v]`;
    sh(FFMPEG, ['-y', '-loglevel', 'error', '-i', mainVid, '-i', popupInfo.video, '-filter_complex', fc, '-map', '[v]', ...enc]);
    const dur = probeDur(mp4);
    console.log('Timeline (video seconds):');
    for (const [l, a, b] of tl) console.log(`  ${(a + off).toFixed(1).padStart(5)}-${(b + off).toFixed(1).padStart(5)}  ${l}`);
    console.log(`PDF tab: ${popupInfo.url} shown ${A.toFixed(1)}-${C.toFixed(1)}s`);
    console.log('Frames (video seconds):', Object.entries(shots).map(([k, v]) => `${k}@${(v + off).toFixed(1)}`).join(', '));
    console.log(`Video duration (ffprobe): ${dur}s -> ${mp4}`);
    if (dur >= 75) throw new Error(`video is ${dur}s, over the 75 s budget`);
    if (!process.env.DEMO_KEEP_RAW) fs.rmSync(RAW, { recursive: true, force: true });
  } finally {
    if (ctx) await ctx.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    if (setupDone) {
      let st = null;
      try { st = await status(); } catch (e) { console.error('status check failed:', e.message); }
      const same = dataDigest(path.join(ROOT, 'data')) === before;
      if (!same || !st || st.stale || st.num_documents !== 3) {
        console.log('Restoring data/ and index/ from backup');
        for (const d of ['data', 'index']) {
          fs.rmSync(path.join(ROOT, d), { recursive: true, force: true });
          sh('cp', ['-a', path.join(BACKUP, d), path.join(ROOT, d)]);
        }
        st = await status().catch(() => null);
      }
      console.log(`Final status: ${st ? `${st.num_documents} documents, stale=${st.stale}` : 'unavailable'}`);
    }
  }
})().catch((e) => { console.error(e); process.exitCode = 1; });
