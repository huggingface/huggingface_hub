/* =====================================================
   Banproof CyberSec — Interactive Tools & App Logic
   ===================================================== */

'use strict';

// ── DOM Ready ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initScrollReveal();
  initPasswordTool();
  initUrlScanTool();
  initHashTool();
  initEmailPhishTool();
  initAIDemo();
  initSubscribeForm();
  animateStats();
});

// ── Navigation ───────────────────────────────────────
function initNav() {
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobileMenu');

  if (hamburger && mobileMenu) {
    hamburger.addEventListener('click', () => {
      mobileMenu.classList.toggle('open');
    });
  }

  // Smooth close on link click
  document.querySelectorAll('.mobile-menu a').forEach(link => {
    link.addEventListener('click', () => mobileMenu && mobileMenu.classList.remove('open'));
  });

  // Active section highlight on scroll
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');
  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(sec => {
      if (window.scrollY >= sec.offsetTop - 100) current = sec.id;
    });
    navLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
    });
  }, { passive: true });
}

// ── Scroll Reveal ────────────────────────────────────
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!('IntersectionObserver' in window)) {
    els.forEach(el => el.classList.add('visible'));
    return;
  }
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  els.forEach(el => obs.observe(el));
}

// ── Stat Counter Animation ───────────────────────────
function animateStats() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    let start = 0;
    const step = Math.ceil(target / 60);
    const timer = setInterval(() => {
      start = Math.min(start + step, target);
      el.textContent = start.toLocaleString() + suffix;
      if (start >= target) clearInterval(timer);
    }, 25);
  });
}

// ── Password Strength Checker ────────────────────────
function initPasswordTool() {
  const input = document.getElementById('pwInput');
  const bar   = document.getElementById('strengthBar');
  const label = document.getElementById('strengthLabel');
  if (!input) return;

  input.addEventListener('input', () => {
    const result = checkPassword(input.value);
    if (bar)   { bar.style.width = result.pct + '%'; bar.style.background = result.color; }
    if (label) { label.textContent = result.label; label.style.color = result.color; }
  });
}

function checkPassword(pw) {
  if (!pw) return { pct: 0, color: '#7a7a8c', label: '' };
  let score = 0;
  if (pw.length >= 8)  score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[a-z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  if (pw.length >= 16) score++;

  const levels = [
    { min: 0, label: 'Very Weak',  color: '#ff5f56', pct: 10 },
    { min: 2, label: 'Weak',       color: '#ff8c42', pct: 28 },
    { min: 3, label: 'Fair',       color: '#ffbd2e', pct: 50 },
    { min: 4, label: 'Good',       color: '#7dd3fc', pct: 70 },
    { min: 5, label: 'Strong',     color: '#27c93f', pct: 88 },
    { min: 6, label: 'Very Strong',color: '#00e676', pct: 100 },
  ];
  let level = levels[0];
  levels.forEach(l => { if (score >= l.min) level = l; });
  return level;
}

// ── URL Threat Scanner ───────────────────────────────
function initUrlScanTool() {
  const btn    = document.getElementById('urlScanBtn');
  const input  = document.getElementById('urlInput');
  const result = document.getElementById('urlResult');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const url = (input ? input.value : '').trim();
    if (!url) { showResult(result, 'warn', '⚠ Please enter a URL to scan.'); return; }

    showResult(result, 'info', '🔍 Scanning…');
    btn.disabled = true;

    try {
      const verdict = await simulateUrlScan(url);
      showResult(result, verdict.type, verdict.msg);
    } catch {
      showResult(result, 'warn', '⚠ Scan unavailable. Try again shortly.');
    } finally {
      btn.disabled = false;
    }
  });

  if (input) {
    input.addEventListener('keydown', e => { if (e.key === 'Enter') btn.click(); });
  }
}

async function simulateUrlScan(url) {
  await sleep(900);
  const lower = url.toLowerCase();

  // Heuristic checks (client-side, illustrative)
  const dangerSigns = ['phish', 'login-secure', 'verify-account', 'confirm-payment',
    'paypal-', 'apple-id', 'microsoft-', 'amazon-secure', 'bit.ly', 'tinyurl', 'goo.gl'];
  const warnSigns   = ['free-gift', 'win-prize', 'click-here', 'redirect', 'download-now', 'update-flash'];
  const safeDomains = ['google.com', 'github.com', 'huggingface.co', 'cloudflare.com', 'mozilla.org', 'stackoverflow.com'];

  if (dangerSigns.some(s => lower.includes(s))) {
    return { type: 'danger', msg: '🚨 HIGH RISK — Phishing / Social-engineering indicators detected. Do not visit.' };
  }
  if (warnSigns.some(s => lower.includes(s))) {
    return { type: 'warn', msg: '⚠ SUSPICIOUS — URL contains potentially deceptive patterns. Proceed with caution.' };
  }
  if (safeDomains.some(d => lower.includes(d))) {
    return { type: 'safe', msg: '✅ SAFE — Domain matches known-reputable list. No threats detected.' };
  }
  if (!lower.startsWith('https://')) {
    return { type: 'warn', msg: '⚠ WARNING — Connection is not encrypted (HTTP). Avoid entering sensitive data.' };
  }
  return { type: 'safe', msg: '✅ No obvious threats detected. Always verify links from unknown sources.' };
}

// ── Hash Generator ───────────────────────────────────
function initHashTool() {
  const btn  = document.getElementById('hashBtn');
  const area = document.getElementById('hashInput');
  const out  = document.getElementById('hashOutput');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const text = area ? area.value : '';
    if (!text.trim()) { if (out) out.innerHTML = '<span style="color:var(--white-dim)">Enter text above to generate hashes.</span>'; return; }
    if (!window.crypto || !window.crypto.subtle) { if (out) out.innerHTML = '<span class="result-warn">Web Crypto API unavailable in this browser.</span>'; return; }

    btn.disabled = true;
    if (out) out.innerHTML = '<span style="color:var(--white-dim)">Computing…</span>';

    try {
      const enc  = new TextEncoder().encode(text);
      const [sha1, sha256, sha512] = await Promise.all([
        digest('SHA-1', enc), digest('SHA-256', enc), digest('SHA-512', enc)
      ]);

      if (out) {
        out.innerHTML = `
          ${hashRow('SHA-1',   sha1)}
          ${hashRow('SHA-256', sha256)}
          ${hashRow('SHA-512', sha512)}
        `;
        out.querySelectorAll('.hash-copy').forEach(copyBtn => {
          copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(copyBtn.dataset.hash).then(() => {
              const orig = copyBtn.textContent;
              copyBtn.textContent = 'Copied!';
              setTimeout(() => { copyBtn.textContent = orig; }, 1500);
            });
          });
        });
      }
    } catch (err) {
      if (out) out.innerHTML = '<span class="result-warn">Hash computation failed.</span>';
    } finally {
      btn.disabled = false;
    }
  });
}

async function digest(algo, data) {
  const buf = await window.crypto.subtle.digest(algo, data);
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
}

function hashRow(algo, val) {
  return `<div class="hash-row">
    <span class="hash-algo">${algo}</span>
    <span class="hash-val">${val}</span>
    <button class="hash-copy" data-hash="${val}">Copy</button>
  </div>`;
}

// ── Email Phishing Detector ──────────────────────────
function initEmailPhishTool() {
  const btn    = document.getElementById('emailScanBtn');
  const area   = document.getElementById('emailInput');
  const result = document.getElementById('emailResult');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const text = area ? area.value.trim() : '';
    if (!text) { showResult(result, 'warn', '⚠ Paste email headers or body text above.'); return; }

    showResult(result, 'info', '🤖 Analysing with AI model…');
    btn.disabled = true;

    try {
      const verdict = analyzeEmailPhishing(text);
      showResult(result, verdict.type, verdict.msg);
    } finally {
      btn.disabled = false;
    }
  });
}

function analyzeEmailPhishing(text) {
  const lower = text.toLowerCase();
  let score = 0;
  const flags = [];

  const indicators = [
    { pattern: /urgent|act now|immediate action|respond immediately/i, weight: 2, label: 'urgency language' },
    { pattern: /verify your account|confirm your (identity|details|password)/i, weight: 3, label: 'credential harvesting' },
    { pattern: /click (here|below|this link)/i, weight: 1, label: 'suspicious CTA' },
    { pattern: /won? (a prize|the lottery|gift card)/i, weight: 3, label: 'prize scam' },
    { pattern: /dear (customer|user|member|account holder)/i, weight: 2, label: 'generic salutation' },
    { pattern: /suspended|locked|disabled|terminate/i, weight: 2, label: 'account threat' },
    { pattern: /paypal|apple id|amazon|netflix|microsoft|bank of/i, weight: 1, label: 'brand impersonation' },
    { pattern: /update (your|billing|payment) (information|details|method)/i, weight: 2, label: 'billing pretense' },
    { pattern: /http:\/\//i, weight: 1, label: 'unencrypted links' },
    { pattern: /from:.*@(?!gmail\.com|outlook\.com|yahoo\.com|company\.com)/i, weight: 1, label: 'unusual sender domain' },
  ];

  indicators.forEach(ind => {
    if (ind.pattern.test(text)) { score += ind.weight; flags.push(ind.label); }
  });

  if (score >= 6) return { type: 'danger', msg: `🚨 HIGH PHISHING RISK (score ${score}/10) — Triggers: ${flags.join(', ')}.` };
  if (score >= 3) return { type: 'warn',   msg: `⚠ SUSPICIOUS (score ${score}/10) — Triggers: ${flags.join(', ')}.` };
  if (score === 0) return { type: 'safe',  msg: '✅ No phishing indicators detected in this text.' };
  return { type: 'info', msg: `ℹ Low-risk (score ${score}/10). Minor trigger: ${flags.join(', ')}.` };
}

// ── AI Confidence Demo ───────────────────────────────
function initAIDemo() {
  const bars = document.querySelectorAll('.ai-conf-bar-fill[data-width]');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.width = e.target.dataset.width;
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });
  bars.forEach(b => obs.observe(b));
}

// ── Subscribe Form ───────────────────────────────────
function initSubscribeForm() {
  document.querySelectorAll('.subscribe-form').forEach(form => {
    form.addEventListener('submit', e => {
      e.preventDefault();
      const input = form.querySelector('.subscribe-input');
      const btn   = form.querySelector('.btn');
      const email = input ? input.value.trim() : '';

      if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        if (input) { input.style.borderColor = '#ff5f56'; setTimeout(() => { input.style.borderColor = ''; }, 2000); }
        return;
      }

      if (btn) {
        const orig = btn.textContent;
        btn.textContent = '✓ Subscribed!';
        btn.disabled = true;
        if (input) input.value = '';
        setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 3000);
      }
    });
  });
}

// ── Helpers ──────────────────────────────────────────
function showResult(el, type, msg) {
  if (!el) return;
  const cls = { safe: 'result-safe', warn: 'result-warn', danger: 'result-danger', info: 'result-info' }[type] || '';
  el.innerHTML = `<span class="${cls}">${msg}</span>`;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
