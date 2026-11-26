/* Cascade chat — same-origin /ui-api, SSE streaming, no client-side key. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const els = {
    messages: $("#messages"),
    empty: $("#empty"),
    input: $("#input"),
    form: $("#composer"),
    send: $("#sendBtn"),
    stop: $("#stopBtn"),
    stats: $("#stats"),
    max: $("#maxTokens"),
    maxOut: $("#maxOut"),
    temp: $("#temperature"),
    tempOut: $("#tempOut"),
    clear: $("#clearBtn"),
    conn: $(".conn"),
    connText: $("#connText"),
  };

  const ENDPOINT = "/ui-api/v1/chat/completions";
  const MODELS = "/ui-api/v1/models";
  const state = {
    messages: [],
    controller: null,
    busy: false,
    lastStats: "",
  };

  /* ---------------- side drawer: 3s idle auto-collapse ---------------- */
  const sideToggle = $("#sideToggle");
  let sideOpen = false;
  let sideTimer = 0;
  function setSide(open) {
    sideOpen = open;
    document.body.classList.toggle("side-open", open);
    sideToggle.setAttribute("aria-expanded", String(open));
    sideToggle.textContent = open ? "收起" : "参数";
    clearTimeout(sideTimer);
    if (open) sideTimer = setTimeout(() => setSide(false), 3000);
  }
  const pokeSide = () => {
    if (!sideOpen) return;
    clearTimeout(sideTimer);
    sideTimer = setTimeout(() => setSide(false), 3000);
  };
  ["mousemove", "mousedown", "keydown", "wheel", "touchstart", "scroll"].forEach((ev) =>
    window.addEventListener(ev, pokeSide, { passive: true })
  );
  sideToggle.addEventListener("click", () => setSide(!sideOpen));
  setSide(true);

  /* ---------------- settings ---------------- */
  const saved = (() => {
    try { return JSON.parse(localStorage.getItem("cascade-chat") || "{}"); }
    catch { return {}; }
  })();
  if (saved.max) els.max.value = saved.max;
  if (saved.temp) els.temp.value = saved.temp;
  const syncSettings = () => {
    els.maxOut.textContent = els.max.value;
    els.tempOut.textContent = Number(els.temp.value).toFixed(1);
    localStorage.setItem(
      "cascade-chat",
      JSON.stringify({ max: els.max.value, temp: els.temp.value })
    );
  };
  els.max.addEventListener("input", syncSettings);
  els.temp.addEventListener("input", syncSettings);
  syncSettings();

  /* ---------------- health ---------------- */
  fetch(MODELS, { headers: { Accept: "application/json" } })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then(() => {
      els.conn.classList.add("online");
      els.connText.textContent = "已连接";
    })
    .catch(() => {
      els.conn.classList.add("offline");
      els.connText.textContent = "连接失败";
    });

  /* ---------------- markdown-lite ---------------- */
  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  function renderMarkdown(text) {
    const parts = String(text).split("```");
    let out = "";
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (i % 2 === 1) {
        const lines = part.replace(/^\n/, "").split("\n");
        const lang = (lines[0] || "").trim();
        const code = lines.slice(lang ? 1 : 0).join("\n").replace(/\n$/, "");
        out += `<pre data-lang="${escapeHtml(lang)}"><code>${escapeHtml(code)}</code></pre>`;
        continue;
      }
      const html = escapeHtml(part)
        .replace(/`([^`\n]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
      out += html
        .split(/\n{2,}/)
        .map((p) => (p.trim() ? `<p>${p.replace(/\n/g, "<br>")}</p>` : ""))
        .join("");
    }
    return out || "";
  }

  /* ---------------- rendering ---------------- */
  function addMessage(role, content, pending = false) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}${pending ? " pending" : ""}`;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "ME" : "AI";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const label = document.createElement("div");
    label.className = "role";
    label.textContent = role === "user" ? "你" : "Cascade";
    const body = document.createElement("div");
    body.className = "content";
    if (role === "user") {
      body.textContent = content;
    } else {
      body.innerHTML = renderMarkdown(content);
    }
    bubble.append(label, body);
    wrap.append(avatar, bubble);
    els.messages.appendChild(wrap);
    if (els.empty) els.empty.style.display = "none";
    return { wrap, body };
  }

  function nearBottom() {
    const el = els.messages;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }
  function scrollToBottom(force = false) {
    if (force || nearBottom()) {
      els.messages.scrollTop = els.messages.scrollHeight;
    }
  }

  /* ---------------- UI state ---------------- */
  function setBusy(v) {
    state.busy = v;
    els.send.disabled = v;
    els.stop.hidden = !v;
    els.input.disabled = v;
    if (!v) els.input.focus();
  }

  function clearChat() {
    if (state.controller) state.controller.abort();
    state.messages = [];
    els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
    if (els.empty) els.empty.style.display = "";
    els.stats.textContent = "";
    state.lastStats = "";
  }

  /* ---------------- streaming send ---------------- */
  async function send(text) {
    text = text.trim();
    if (!text || state.busy) return;

    state.messages.push({ role: "user", content: text });
    addMessage("user", text);
    els.input.value = "";
    els.input.style.height = "auto";
    scrollToBottom(true);

    const assistant = addMessage("assistant", "", true);
    let content = "";
    let chunks = 0;
    let ttft = 0;
    const t0 = performance.now();
    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);
    els.stats.textContent = "思考中…";

    try {
      const resp = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          model: "cascade",
          messages: [
            {
              role: "system",
              content:
                "你是 Cascade 助手。请用中文直接回答用户的问题；不要续写对话、不要模拟用户，也不要输出 User: / Assistant: 这类角色标记。",
            },
            ...state.messages,
          ],
          max_tokens: Number(els.max.value),
          temperature: Number(els.temp.value),
          stop: ["User:", "Assistant:", "用户:", "助手:"],
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const detail = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status}${detail ? ` · ${detail.slice(0, 180)}` : ""}`);
      }
      if (!resp.body) throw new Error("当前浏览器不支持流式响应");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let done = false;

      while (!done) {
        const { value, done: eof } = await reader.read();
        if (eof) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        let idx;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const event = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 2);
          for (const line of event.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            if (payload === "[DONE]") { done = true; break; }
            let chunk;
            try { chunk = JSON.parse(payload); } catch { continue; }
            if (chunk.error) throw new Error(chunk.error.message || "engine error");
            const delta = chunk.choices?.[0]?.delta?.content || "";
            if (!delta) continue;
            if (!ttft) ttft = performance.now() - t0;
            content += delta;
            chunks += 1;
            // 基座模型可能续写角色标记；即使服务端 stop 失效也在这里截断。
            const roleMarkers = ["User:", "Assistant:", "用户:", "助手:"];
            let cut = -1;
            for (const marker of roleMarkers) {
              const at = content.indexOf(marker);
              if (at >= 0) cut = cut < 0 ? at : Math.min(cut, at);
            }
            if (cut >= 0) {
              content = content.slice(0, cut).trimEnd();
              assistant.body.innerHTML = renderMarkdown(content);
              controller.abort();
              done = true;
              break;
            }
            assistant.body.innerHTML = renderMarkdown(content);
            scrollToBottom();
            const elapsed = (performance.now() - t0) / 1000;
            state.lastStats =
              `TTFT ${Math.round(ttft)}ms · ${chunks} chunks · ${elapsed.toFixed(1)}s`;
            els.stats.textContent = state.lastStats;
          }
          if (done) break;
        }
      }

      assistant.wrap.classList.remove("pending");
      for (const tail of ["\nUser", "\nAssistant", "\n用户", "\n助手"]) {
        if (content.endsWith(tail)) content = content.slice(0, -tail.length).trimEnd();
      }
      assistant.body.innerHTML = renderMarkdown(content);
      if (!content.trim()) {
        assistant.body.innerHTML = '<div class="error-box">模型没有返回内容。</div>';
      } else {
        state.messages.push({ role: "assistant", content });
      }
    } catch (err) {
      assistant.wrap.classList.remove("pending");
      if (err.name === "AbortError") {
        if (!content) assistant.body.innerHTML = '<div class="error-box">已停止生成。</div>';
        else state.messages.push({ role: "assistant", content });
      } else {
        assistant.body.innerHTML = `<div class="error-box">请求失败：${escapeHtml(
          String(err.message || err)
        )}</div>`;
      }
    } finally {
      state.controller = null;
      setBusy(false);
      scrollToBottom();
    }
  }

  /* ---------------- events ---------------- */
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(els.input.value);
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(els.input.value);
    }
  });

  els.input.addEventListener("input", () => {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(180, els.input.scrollHeight)}px`;
  });

  els.stop.addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });

  els.clear.addEventListener("click", clearChat);

  document.querySelectorAll(".suggest, .empty-chips button").forEach((btn) => {
    btn.addEventListener("click", () => {
      els.input.value = btn.textContent.trim();
      send(els.input.value);
    });
  });

  els.input.focus();
})();
