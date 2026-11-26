/* Cascade landing — interactions: nav, hero rotation, lazy media,
   scroll reveals, count-up, parallax, code tabs. No dependencies. */
(() => {
  "use strict";

  window.__cascadeApp = "1";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* ---------------- nav + progress ---------------- */
  const nav = $("#nav");
  const progress = $("#scrollProgress");
  let lastY = window.scrollY;

  function onScroll() {
    const y = window.scrollY;
    loadVisibleLazy();
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = `${max > 0 ? (y / max) * 100 : 0}%`;
    if (nav) nav.classList.toggle("is-scrolled", y > 24);
    if (!reduced) parallax(y);
    lastY = y;
  }
  window.addEventListener("scroll", onScroll, { passive: true });

  const toggle = $("#navToggle");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    $$(".nav-links a").forEach((a) =>
      a.addEventListener("click", () => {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      })
    );
  }

  /* ---------------- hero: crossfade large images ---------------- */
  const slides = $$(".hero-slide");
  slides.forEach((el) => {
    const bg = el.dataset.bg;
    if (bg) el.style.backgroundImage = `url("${bg}")`;
  });
  if (!reduced && slides.length > 1) {
    let idx = 0;
    setInterval(() => {
      slides[idx].classList.remove("is-active");
      idx = (idx + 1) % slides.length;
      slides[idx].classList.add("is-active");
    }, 6200);
  }

  /* ---------------- parallax ---------------- */
  const parallaxEls = $$("[data-parallax]");
  let ticking = false;
  function parallax(y) {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      parallaxEls.forEach((el) => {
        const speed = parseFloat(el.dataset.parallax || "0");
        el.style.transform = `translate3d(0, ${(y * speed).toFixed(1)}px, 0)`;
      });
      ticking = false;
    });
  }

  /* ---------------- lazy images ---------------- */
  const lazyImgs = $$("img.lazy");
  function loadLazy(img) {
    const src = img.dataset.src;
    if (!src) return;
    img.src = src;
    img.removeAttribute("data-src");
    const done = () => img.classList.add("is-loaded");
    img.addEventListener("load", done, { once: true });
    if (img.complete && img.naturalWidth > 0) done();
  }
  // 兜底：不依赖 IO（无头/兼容性/快速跳转）——滚动时按距离补载
  function loadVisibleLazy() {
    const vh = window.innerHeight || 800;
    lazyImgs.forEach((img) => {
      if (!img.dataset.src) return;
      const r = img.getBoundingClientRect();
      if (r.top < vh + 700 && r.bottom > -700) loadLazy(img);
    });
  }
  const imgObserver = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        loadLazy(entry.target);
        obs.unobserve(entry.target);
      });
    },
    { rootMargin: "300px 0px", threshold: 0.01 }
  );
  lazyImgs.forEach((img) => imgObserver.observe(img));

  /* ---------------- reveal on scroll ---------------- */
  const revealEls = $$(".reveal");
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
        if (entry.target.classList.contains("metric-card")) startBars(entry.target);
      });
    },
    { threshold: 0.16, rootMargin: "0px 0px -8% 0px" }
  );
  revealEls.forEach((el) => revealObserver.observe(el));

  function startBars(card) {
    const bar = card.querySelector(".bar span");
    if (bar) bar.style.width = bar.style.getPropertyValue("--w") || "60%";
  }

  /* ---------------- count-up ---------------- */
  function animateCount(el) {
    const to = parseFloat(el.dataset.to || "0");
    const dec = parseInt(el.dataset.dec || "0", 10);
    if (reduced) {
      el.textContent = to.toFixed(dec);
      return;
    }
    const dur = 1500;
    const t0 = performance.now();
    const from = 0;
    function tick(now) {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = from + (to - from) * eased;
      el.textContent = dec > 0 ? val.toFixed(dec) : Math.round(val).toString();
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = to.toFixed(dec);
    }
    requestAnimationFrame(tick);
  }

  const counters = $$(".count");
  const countObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        animateCount(entry.target);
        countObserver.unobserve(entry.target);
      });
    },
    { threshold: 0.6 }
  );
  counters.forEach((el) => countObserver.observe(el));

  /* ---------------- code tabs + copy ---------------- */
  const tabs = $$(".code-tab");
  const panes = $$(".code-pane");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.tab;
      tabs.forEach((t) => {
        const on = t === tab;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", String(on));
      });
      panes.forEach((p) => p.classList.toggle("is-active", p.dataset.pane === key));
    });
  });

  const copyBtn = $("#copyBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const active = $(".code-pane.is-active");
      const text = active ? active.innerText : "";
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      copyBtn.textContent = "已复制";
      copyBtn.classList.add("is-done");
      setTimeout(() => {
        copyBtn.textContent = "复制";
        copyBtn.classList.remove("is-done");
      }, 1800);
    });
  }

  /* ---------------- active nav link ---------------- */
  const sections = $$("main section[id]");
  const navLinks = $$(".nav-links a");
  if (sections.length && navLinks.length) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const id = entry.target.id;
          navLinks.forEach((a) =>
            a.classList.toggle("is-active", a.getAttribute("href") === `#${id}`)
          );
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    sections.forEach((s) => spy.observe(s));
  }

  onScroll();
  window.addEventListener("load", onScroll, { once: true });
  loadVisibleLazy();
})();