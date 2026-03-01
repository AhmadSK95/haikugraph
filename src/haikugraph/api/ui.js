    const STORAGE_WORKSPACE = "datada_workspace_v3";
    const STORAGE_CONN = "datada_conn";
    const STORAGE_MODE = "datada_mode";
    const STORAGE_MODEL_SELECTIONS = "datada_model_selections_v1";

    const state = {
      workspace: loadWorkspace(),
      connectionId: localStorage.getItem(STORAGE_CONN) || "default",
      mode: localStorage.getItem(STORAGE_MODE) || "auto",
      providers: {},
      modelSelections: loadModelSelections(),
      modelCatalog: {
        local: {available: false, options: [], reason: "loading"},
        openai: {available: false, options: [], reason: "loading"},
        anthropic: {available: false, options: [], reason: "loading"},
      },
      fixContext: null,
      ruleCache: [],
    };

    function loadWorkspace() {
      try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_WORKSPACE) || "{}");
        if (parsed && parsed.projects && parsed.currentProjectId && parsed.currentThreadId) return parsed;
      } catch {}
      const projectId = uid();
      const threadId = uid();
      return {
        projects: [{
          id: projectId,
          name: "Default Project",
          threads: [{
            id: threadId,
            name: "Thread 1",
            sessionId: uid(),
            turns: [],
            createdAt: Date.now()
          }]
        }],
        currentProjectId: projectId,
        currentThreadId: threadId
      };
    }

    function saveWorkspace() {
      localStorage.setItem(STORAGE_WORKSPACE, JSON.stringify(state.workspace));
      localStorage.setItem(STORAGE_CONN, state.connectionId || "default");
      localStorage.setItem(STORAGE_MODE, state.mode || "auto");
      localStorage.setItem(STORAGE_MODEL_SELECTIONS, JSON.stringify(state.modelSelections || {}));
    }

    function loadModelSelections() {
      try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_MODEL_SELECTIONS) || "{}");
        if (parsed && typeof parsed === "object") return parsed;
      } catch {}
      return {local: "", openai: "", anthropic: ""};
    }

    function uid() {
      return crypto.randomUUID ? crypto.randomUUID() : ("id_" + Math.random().toString(36).slice(2, 12));
    }

    function $(id) { return document.getElementById(id); }
    function esc(v) {
      const d = document.createElement("div");
      d.textContent = v == null ? "" : String(v);
      return d.innerHTML;
    }
    function fmt(n) {
      if (n == null) return "—";
      if (typeof n === "number") return n.toLocaleString();
      return String(n);
    }

    const ADMIN_HEADERS = Object.freeze({
      "x-datada-role": "admin",
      "x-datada-tenant-id": "public",
    });
    const ADMIN_JSON_HEADERS = Object.freeze({
      "Content-Type": "application/json",
      "x-datada-role": "admin",
      "x-datada-tenant-id": "public",
    });

    function asRecord(payload, context) {
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error(`Malformed payload for ${context}.`);
      }
      return payload;
    }

    async function apiFetchJson(path, {method = "GET", headers = {}, body, timeoutMs = 30000} = {}) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(path, {
          method,
          headers,
          body: body == null ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
        const text = await response.text();
        let payload = {};
        if (text) {
          try {
            payload = JSON.parse(text);
          } catch {
            throw new Error(`Invalid JSON response from ${path}.`);
          }
        }
        if (!response.ok) {
          const detail = payload && typeof payload === "object"
            ? (payload.detail || payload.message || payload.error)
            : "";
          throw new Error(String(detail || `${response.status} ${response.statusText}`));
        }
        return asRecord(payload, path);
      } catch (err) {
        if (err && err.name === "AbortError") {
          throw new Error(`Request timed out for ${path}.`);
        }
        throw err;
      } finally {
        clearTimeout(timeout);
      }
    }

    const apiClient = {
      async clearSession(body) {
        return apiFetchJson("/api/assistant/session/clear", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 12000,
        });
      },
      async startAsyncQuery(body) {
        const payload = await apiFetchJson("/api/assistant/query/async", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 20000,
        });
        const jobId = String(payload.job_id || "").trim();
        if (!jobId) throw new Error("Malformed async response: missing job_id.");
        return {job_id: jobId};
      },
      async getAsyncQueryStatus(jobId) {
        const payload = await apiFetchJson(`/api/assistant/query/async/${encodeURIComponent(jobId)}`, {
          headers: ADMIN_HEADERS,
          timeoutMs: 30000,
        });
        const status = String(payload.status || "").toLowerCase();
        if (!status) throw new Error("Malformed async status payload: missing status.");
        return {
          status,
          response: payload.response && typeof payload.response === "object" ? payload.response : null,
          error: payload.error ? String(payload.error) : "",
        };
      },
      async submitFix(body) {
        return apiFetchJson("/api/assistant/fix", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 25000,
        });
      },
      async upsertRule(path, body) {
        return apiFetchJson(path, {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 20000,
        });
      },
      async setRuleStatus(body) {
        return apiFetchJson("/api/assistant/rules/status", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 15000,
        });
      },
      async rollbackRule(body) {
        return apiFetchJson("/api/assistant/rules/rollback", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 15000,
        });
      },
      async listRules(connectionId, limit = 300) {
        const payload = await apiFetchJson(
          `/api/assistant/rules?db_connection_id=${encodeURIComponent(connectionId)}&limit=${encodeURIComponent(String(limit))}`,
          {headers: ADMIN_HEADERS, timeoutMs: 20000},
        );
        payload.rules = Array.isArray(payload.rules) ? payload.rules : [];
        return payload;
      },
      async listConnections() {
        const payload = await apiFetchJson("/api/assistant/connections", {headers: ADMIN_HEADERS, timeoutMs: 15000});
        payload.connections = Array.isArray(payload.connections) ? payload.connections : [];
        return payload;
      },
      async providers() {
        const payload = await apiFetchJson("/api/assistant/providers", {headers: ADMIN_HEADERS, timeoutMs: 15000});
        payload.checks = (payload.checks && typeof payload.checks === "object") ? payload.checks : {};
        return payload;
      },
      async health() {
        return apiFetchJson("/api/assistant/health", {timeoutMs: 12000});
      },
      async modelCatalog(provider) {
        return apiFetchJson(`/api/assistant/models/${encodeURIComponent(provider)}`, {timeoutMs: 15000});
      },
      async pullLocalModel(body) {
        return apiFetchJson("/api/assistant/models/local/pull", {
          method: "POST",
          headers: ADMIN_JSON_HEADERS,
          body,
          timeoutMs: 60000,
        });
      },
    };

    function md(raw) {
      if (!raw) return "";
      return String(raw)
        .replace(/^### (.+)$/gm, "<h3>$1</h3>")
        .replace(/^## (.+)$/gm, "<h2>$1</h2>")
        .replace(/^[-*] (.+)$/gm, "<li>$1</li>")
        .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\n{2,}/g, "</p><p>")
        .replace(/^(?!<[hulo])(.+)$/gm, "<p>$1</p>")
        .replace(/<ul>\s*<ul>/g, "<ul>")
        .replace(/<\/ul>\s*<\/ul>/g, "</ul>");
    }

    function currentProject() {
      return state.workspace.projects.find(p => p.id === state.workspace.currentProjectId) || state.workspace.projects[0];
    }

    function currentThread() {
      const p = currentProject();
      return p.threads.find(t => t.id === state.workspace.currentThreadId) || p.threads[0];
    }

    function renameProject() {
      const p = currentProject();
      const val = $("projectName").value.trim();
      if (!val) return toast("Project name cannot be empty.");
      p.name = val;
      saveWorkspace();
      renderSidebar();
      toast("Project name updated.");
    }

    function newThread() {
      const p = currentProject();
      const nextIdx = p.threads.length + 1;
      const thread = {
        id: uid(),
        name: "Thread " + nextIdx,
        sessionId: uid(),
        turns: [],
        createdAt: Date.now(),
      };
      p.threads.unshift(thread);
      state.workspace.currentThreadId = thread.id;
      saveWorkspace();
      renderAll();
      toast("New thread created.");
    }

    function duplicateThread() {
      const src = currentThread();
      const p = currentProject();
      const copy = JSON.parse(JSON.stringify(src));
      copy.id = uid();
      copy.name = src.name + " Copy";
      copy.sessionId = uid();
      copy.createdAt = Date.now();
      p.threads.unshift(copy);
      state.workspace.currentThreadId = copy.id;
      saveWorkspace();
      renderAll();
      toast("Thread duplicated.");
    }

    function selectThread(threadId) {
      state.workspace.currentThreadId = threadId;
      saveWorkspace();
      renderAll();
    }

    async function clearCurrentThread() {
      const thread = currentThread();
      if (!thread) return;
      thread.turns = [];
      try {
        await apiClient.clearSession({
          db_connection_id: state.connectionId || "default",
          session_id: thread.sessionId,
          tenant_id: "public",
        });
      } catch {}
      saveWorkspace();
      renderThread();
      toast("Thread cleared.");
    }

    function renderSidebar() {
      const p = currentProject();
      $("projectName").value = p.name;
      const list = $("threadList");
      list.innerHTML = p.threads.map(t => {
        const active = t.id === state.workspace.currentThreadId ? "active" : "";
        const turns = (t.turns || []).length;
        const when = t.createdAt ? new Date(t.createdAt).toLocaleDateString() : "";
        return `
          <button class="thread-item ${active}" onclick="selectThread('${esc(t.id)}')">
            <span class="thread-name">${esc(t.name)}</span>
            <span class="thread-meta">${turns} turn${turns===1?"":"s"} • ${esc(when)}</span>
          </button>
        `;
      }).join("");
      $("sessionLabel").textContent = currentThread().sessionId.slice(0, 12);
    }

    function detectChartType(columns, rows) {
      if (!columns || !rows || rows.length < 2 || columns.length < 2) return null;
      const numericCols = columns.filter(c => rows.some(r => typeof r[c] === "number"));
      const textCols = columns.filter(c => !numericCols.includes(c));
      if (!numericCols.length || !textCols.length) return null;
      const labelCol = textCols[0];
      const isTimeLike = /date|month|year|week|day|time|period/i.test(labelCol);
      return {type: isTimeLike ? "line" : "bar", labelCol, valueCols: numericCols.slice(0, 3)};
    }

    let chartCounter = 0;
    function buildChart(columns, rows) {
      const spec = detectChartType(columns, rows);
      if (!spec) return "";
      const id = "chart_" + (++chartCounter);
      setTimeout(() => {
        const el = document.getElementById(id);
        if (!el) return;
        const labels = rows.map(r => String(r[spec.labelCol] ?? ""));
        const palette = [
          {b:"rgba(77,216,179,.9)", a:"rgba(77,216,179,.2)"},
          {b:"rgba(130,170,255,.9)", a:"rgba(130,170,255,.2)"},
          {b:"rgba(244,201,93,.9)", a:"rgba(244,201,93,.2)"}
        ];
        const datasets = spec.valueCols.map((col, i) => ({
          label: col,
          data: rows.map(r => r[col]),
          borderColor: palette[i % palette.length].b,
          backgroundColor: spec.type === "line" ? palette[i % palette.length].a : palette[i % palette.length].b,
          borderWidth: 2,
          tension: .28,
          fill: spec.type === "line",
          borderRadius: spec.type === "bar" ? 4 : 0
        }));
        new Chart(el, {
          type: spec.type,
          data: {labels, datasets},
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                labels: {color: "#9db0d8", font: {size: 11}},
                display: datasets.length > 1
              }
            },
            scales: {
              x: {ticks: {color: "#9db0d8", font: {size: 10}}, grid: {color: "#223865"}},
              y: {ticks: {color: "#9db0d8", font: {size: 10}}, grid: {color: "#223865"}}
            }
          }
        });
      }, 60);
      return `<div class="chart-wrap"><canvas id="${id}"></canvas></div>`;
    }

    function renderThread() {
      const thread = currentThread();
      const turns = thread.turns || [];
      const el = $("thread");
      if (!turns.length) {
        el.innerHTML = `
          <div class="empty">
            Ask a question to start this thread. Try:
            <br />• "What kind of data do I have?"
            <br />• "Total MT103 transactions count split by month and platform"
            <br />• "Top 5 platforms by total transaction amount in December 2025"
          </div>
        `;
        return;
      }
      el.innerHTML = turns.map((turn, idx) => {
        const r = turn.response;
        if (!r) {
          return `
            <div class="turn">
              <div class="user">${esc(turn.goal)}</div>
              <div class="assistant">
                <div class="assistant-body"><p>Analyzing...</p></div>
              </div>
            </div>
          `;
        }
        if (!r.success && r.error) {
          return `
            <div class="turn">
              <div class="user">${esc(turn.goal)}</div>
              <div class="assistant">
                <div class="assistant-body"><p><strong>Query failed:</strong> ${esc(r.error)}</p></div>
                <div class="meta">
                  <span class="chip warn">degraded</span>
                  <div class="meta-actions">
                    <button class="action" onclick="openExplain(${idx})">Explain Yourself</button>
                    <button class="action fix" onclick="openFix(${idx})">Fix</button>
                  </div>
                </div>
              </div>
            </div>
          `;
        }
        const score = Number(r.confidence_score || 0);
        const conf = score >= .75 ? "good" : (score >= .45 ? "warn" : "");
        const mode = (r.runtime && r.runtime.mode) ? r.runtime.mode : state.mode;
        const analysisVersion = String(r.analysis_version || "v1");
        const sliceSignature = String(r.slice_signature || "");
        const truthScore = Number.isFinite(Number(r.truth_score)) ? Number(r.truth_score) : null;
        const qualityFlags = Array.isArray(r.quality_flags) ? r.quality_flags.slice(0, 4) : [];
        const assumptions = Array.isArray(r.assumptions) ? r.assumptions.slice(0, 4) : [];
        const providerEffective = String(r.provider_effective || ((r.runtime || {}).provider || mode || ""));
        const fallback = (r.fallback_used && typeof r.fallback_used === "object") ? r.fallback_used : {};
        const fallbackUsed = Boolean(fallback.used);
        const fallbackReason = String(fallback.reason || "");
        const stageTimings = (r.stage_timings_ms && typeof r.stage_timings_ms === "object") ? r.stage_timings_ms : {};
        const stageTimingTotal = Object.values(stageTimings)
          .map(v => Number(v))
          .filter(v => Number.isFinite(v))
          .reduce((sum, v) => sum + v, 0);
        const sqlPresent = !!(r.sql && String(r.sql).trim());
        const canExplain = sqlPresent || (r.agent_trace || []).length || (r.decision_flow || []).length;
        const suggestions = (r.suggested_questions || []).slice(0, 3);
        const columns = r.columns || [];
        const rows = r.sample_rows || [];
        const governanceNotes = ((((r.data_quality || {}).business_rules || {}).governance_notes) || []).slice(0, 2);
        const table = columns.length && rows.length ? `
          <div class="table-wrap">
            <table>
              <thead><tr>${columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
              <tbody>
                ${rows.slice(0, 20).map(row => `<tr>${columns.map(c => `<td>${fmt(row[c])}</td>`).join("")}</tr>`).join("")}
              </tbody>
            </table>
          </div>
        ` : "";
        const chart = buildChart(columns, rows);
        const preview = (chart || table) ? `
          <details class="data-preview">
            <summary>View data preview (${rows.length} row sample)</summary>
            ${chart}
            ${table}
          </details>
        ` : "";
        return `
          <div class="turn">
            <div class="user">${esc(turn.goal)}</div>
            <div class="assistant">
              <div class="assistant-body">
                ${governanceNotes.length ? `<p><strong>Policy note:</strong> ${esc(governanceNotes.join(" | "))}</p>` : ""}
                ${md(r.answer_markdown || "")}
                ${
                  assumptions.length
                    ? `<details class="data-preview">
                         <summary>Assumptions (${assumptions.length})</summary>
                         <ul style="margin:8px 0 0 18px;padding:0">${assumptions.map(a => `<li style="margin:0 0 6px 0">${esc(String(a))}</li>`).join("")}</ul>
                       </details>`
                    : ""
                }
                ${
                  qualityFlags.length
                    ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">${qualityFlags.map(flag => `<span class="chip warn">${esc(String(flag))}</span>`).join("")}</div>`
                    : ""
                }
                ${preview}
                ${suggestions.length ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">${suggestions.map(q => `<button class="btn small" onclick="injectQuestion('${esc(q)}')">${esc(q)}</button>`).join("")}</div>` : ""}
              </div>
              <div class="meta">
                <span class="chip ${conf}">${Math.round(score*100)}% confidence</span>
                <span class="chip">${Math.round(r.execution_time_ms || 0)} ms</span>
                <span class="chip">${esc(mode)}</span>
                <span class="chip">${esc(analysisVersion)}</span>
                ${truthScore != null ? `<span class="chip ${truthScore >= 90 ? "good" : "warn"}">truth ${Math.round(truthScore)}</span>` : ""}
                ${sliceSignature ? `<span class="chip">slice ${esc(sliceSignature.slice(0, 8))}</span>` : ""}
                ${providerEffective ? `<span class="chip">provider ${esc(providerEffective)}</span>` : ""}
                ${fallbackUsed ? `<span class="chip warn">fallback${fallbackReason ? `: ${esc(fallbackReason.slice(0, 24))}` : ""}</span>` : ""}
                ${stageTimingTotal > 0 ? `<span class="chip">${Math.round(stageTimingTotal)} ms stages</span>` : ""}
                <div class="meta-actions">
                  ${canExplain ? `<button class="action" onclick="openExplain(${idx})">Explain Yourself</button>` : ""}
                  <button class="action fix" onclick="openFix(${idx})">Fix</button>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join("");
      el.scrollTop = 0;
    }

    function renderAll() {
      renderSidebar();
      renderThread();
      saveWorkspace();
    }

    function injectQuestion(q) {
      $("goalInput").value = q;
      $("goalInput").focus();
    }

    async function runQuery() {
      const input = $("goalInput");
      const goal = input.value.trim();
      if (!goal) return;
      const thread = currentThread();
      const runBtn = $("runBtn");
      runBtn.disabled = true;
      input.value = "";
      const turn = {goal, timestamp: Date.now(), response: null};
      thread.turns.unshift(turn);
      renderThread();
      saveWorkspace();

      try {
        const mode = state.mode || "auto";
        const selectedLocalModel = (state.modelSelections.local || "").trim();
        const selectedOpenAIModel = (state.modelSelections.openai || "").trim();
        const selectedAnthropicModel = (state.modelSelections.anthropic || "").trim();
        const requestBody = {
          goal,
          db_connection_id: state.connectionId || "default",
          llm_mode: mode,
          session_id: thread.sessionId,
          storyteller_mode: false,
          auto_correction: true,
          strict_truth: true,
          max_refinement_rounds: 2,
          max_candidate_plans: 5,
          tenant_id: "public",
          role: "admin",
        };
        if (mode === "local" && selectedLocalModel) {
          requestBody.local_model = selectedLocalModel;
          requestBody.local_narrator_model = selectedLocalModel;
        } else if (mode === "openai" && selectedOpenAIModel) {
          requestBody.openai_model = selectedOpenAIModel;
          requestBody.openai_narrator_model = selectedOpenAIModel;
        } else if (mode === "anthropic" && selectedAnthropicModel) {
          requestBody.anthropic_model = selectedAnthropicModel;
          requestBody.anthropic_narrator_model = selectedAnthropicModel;
        }
        const acceptedJson = await apiClient.startAsyncQuery(requestBody);
        const jobId = acceptedJson.job_id;
        const started = Date.now();
        const timeoutMs = 180000;
        while (Date.now() - started < timeoutMs) {
          await new Promise(r => setTimeout(r, 800));
          const statusJson = await apiClient.getAsyncQueryStatus(jobId);
          const status = statusJson.status;
          if (status === "completed") {
            turn.response = statusJson.response || {success:false,error:"Missing payload"};
            break;
          }
          if (status === "failed" || status === "canceled") {
            throw new Error(statusJson.error || `Query ${status}.`);
          }
        }
        if (!turn.response) throw new Error("Query timed out.");
      } catch (err) {
        turn.response = {success:false,error:(err && err.message) ? err.message : "Network error"};
      }
      runBtn.disabled = false;
      renderThread();
      saveWorkspace();
    }

    function closeExplain(ev) {
      if (ev && ev.target && ev.target.id !== "explainOverlay") return;
      $("explainOverlay").classList.remove("open");
    }

    function openExplain(turnIdx) {
      const turn = currentThread().turns[turnIdx];
      if (!turn || !turn.response) return;
      const r = turn.response;
      const explainability = (r.explainability && typeof r.explainability === "object") ? r.explainability : {};
      const businessView = (explainability.business_view && typeof explainability.business_view === "object")
        ? explainability.business_view
        : {};
      const technicalView = (explainability.technical_view && typeof explainability.technical_view === "object")
        ? explainability.technical_view
        : {};
      const flow = Array.isArray(technicalView.decision_flow)
        ? technicalView.decision_flow
        : (Array.isArray(r.decision_flow) ? r.decision_flow : []);
      const trace = Array.isArray(technicalView.agent_trace)
        ? technicalView.agent_trace
        : (Array.isArray(r.agent_trace) ? r.agent_trace : []);
      const checks = Array.isArray(r.sanity_checks) ? r.sanity_checks : [];
      const qualityForEdges = (technicalView.data_quality && typeof technicalView.data_quality === "object")
        ? technicalView.data_quality
        : (r.data_quality || {});
      const edges = ((((qualityForEdges || {}).blackboard || {}).edges) || []);
      const body = $("explainBody");
      const businessSteps = Array.isArray(businessView.plain_steps) ? businessView.plain_steps : [];
      const focus = (businessView.focus && typeof businessView.focus === "object") ? businessView.focus : {};
      const quality = (businessView.quality && typeof businessView.quality === "object") ? businessView.quality : {};

      const timeline = trace.length ? trace.map((step, idx) => {
        const agentName = String(step.agent || step.role || `agent_${idx + 1}`);
        const next = edges
          .filter(e => String(e.from || "") === agentName)
          .slice(0, 5)
          .map(e => `${String(e.artifact_type || "handoff")} -> ${String(e.to || "")}`);
        const pairedFlow = flow[idx] || null;
        const statusColor = step.status === "failed" ? "#ff6f7b" : (step.status === "warning" ? "#f4c95d" : "#6ad28b");
        return `
          <div class="timeline-card">
            <div class="timeline-head">
              <span>${idx + 1}. ${esc(agentName)}</span>
              <span style="color:${statusColor}">${esc(step.status || "success")} • ${Math.round(Number(step.duration_ms || 0))} ms</span>
            </div>
            <div class="timeline-stage">${esc(step.role || pairedFlow?.step || "decision")}</div>
            <div class="timeline-desc">${esc(step.summary || pairedFlow?.description || "No summary available.")}</div>
            ${step.reasoning ? `<div class="agent-text"><strong>Why:</strong> ${esc(step.reasoning)}</div>` : ""}
            ${step.contribution ? `<div class="agent-text"><strong>Output:</strong> ${esc(step.contribution)}</div>` : ""}
            ${
              next.length
                ? `<div class="handoff-row">${next.map(text => `<span class="handoff-chip">${esc(text)}</span>`).join("")}</div>`
                : ""
            }
            ${
              pairedFlow && pairedFlow.details
                ? `<details><summary>Step details</summary><pre>${esc(JSON.stringify(pairedFlow.details, null, 2))}</pre></details>`
                : ""
            }
          </div>
        `;
      }).join("") : `<div class="timeline-card">No trace available.</div>`;

      const checkHtml = checks.length ? checks.map(c => `
        <div class="chip ${c.passed ? "good" : "warn"}">${esc(c.check_name || "")}: ${c.passed ? "pass" : "fail"}</div>
      `).join("") : `<span class="chip">No checks</span>`;
      const businessFocusHtml = [
        focus.intent ? `<span class="chip">Intent: ${esc(String(focus.intent))}</span>` : "",
        focus.domain ? `<span class="chip">Domain: ${esc(String(focus.domain))}</span>` : "",
        focus.metric ? `<span class="chip">Metric: ${esc(String(focus.metric))}</span>` : "",
        focus.time_scope ? `<span class="chip">Time scope: ${esc(String(focus.time_scope))}</span>` : "",
      ].filter(Boolean).join("");
      const businessStepsHtml = businessSteps.length
        ? `<ul style="margin:8px 0 0 18px;padding:0">${businessSteps.map(s => `<li style="margin:0 0 6px 0">${esc(String(s))}</li>`).join("")}</ul>`
        : `<div class="subtle">Business summary unavailable for this response.</div>`;
      const technicalSql = technicalView.sql || r.sql || "(none)";
      const diagnosticsPayload = {
        runtime: technicalView.runtime || r.runtime || {},
        contract_spec: technicalView.contract_spec || r.contract_spec || {},
        contract_validation: technicalView.contract_validation || r.contract_validation || {},
        data_quality: technicalView.data_quality || r.data_quality || {},
        contribution_map: technicalView.contribution_map || r.contribution_map || [],
        analysis_version: r.analysis_version || "v1",
        slice_signature: r.slice_signature || "",
        quality_flags: r.quality_flags || [],
        assumptions: r.assumptions || [],
        truth_score: r.truth_score,
        stage_timings_ms: r.stage_timings_ms || {},
        provider_effective: r.provider_effective || "",
        fallback_used: r.fallback_used || {},
      };

      body.innerHTML = `
        <div class="section">
          <h4>Question + final answer</h4>
          <div class="section-body">
            <div style="margin-bottom:8px"><strong>Question:</strong> ${esc(turn.goal)}</div>
            <div>${md(r.answer_markdown || "")}</div>
          </div>
        </div>
        <div class="section">
          <h4>Business view</h4>
          <div class="section-body">
            <div style="margin-bottom:8px">${esc(String(businessView.answer_summary || "No business summary generated."))}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">${businessFocusHtml || `<span class="chip">No focus metadata</span>`}</div>
            ${businessStepsHtml}
            <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
              <span class="chip">Confidence: ${esc(String(quality.confidence_score != null ? quality.confidence_score : r.confidence_score || 0))}</span>
              <span class="chip ${(quality.contract_valid === false) ? "warn" : "good"}">Contract: ${(quality.contract_valid === false) ? "check warnings" : "valid"}</span>
            </div>
          </div>
        </div>
        <div class="section">
          <h4>Technical drill-down</h4>
          <div class="section-body">
            <h4 style="margin:0 0 8px 0;font-size:13px">Agent decision timeline</h4>
            <div class="timeline-stack">${timeline}</div>
            <h4 style="margin:12px 0 8px 0;font-size:13px">SQL + validation</h4>
            <details open><summary>SQL</summary><pre>${esc(technicalSql)}</pre></details>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px">${checkHtml}</div>
            <details style="margin-top:8px"><summary>Advanced diagnostics (JSON)</summary><pre>${esc(JSON.stringify(diagnosticsPayload, null, 2))}</pre></details>
          </div>
        </div>
      `;
      $("explainOverlay").classList.add("open");
    }

    function openFix(turnIdx) {
      const turn = currentThread().turns[turnIdx];
      if (!turn || !turn.response) return;
      const r = turn.response || {};
      const grounding = (r.data_quality && r.data_quality.grounding) ? r.data_quality.grounding : {};
      const firstWords = String(turn.goal || "").split(/\s+/).filter(Boolean).slice(0, 3).join(" ").toLowerCase();
      state.fixContext = {
        turnIdx,
        goal: turn.goal,
        trace_id: r.trace_id || "",
        session_id: currentThread().sessionId,
      };
      $("fixIssue").value = "";
      $("fixKeyword").value = firstWords || "business term";
      $("fixDomain").value = String(grounding.domain || grounding.table || "").includes("quotes") ? "quotes" : "transactions";
      $("fixTable").value = grounding.table || "";
      $("fixMetric").value = grounding.metric || "";
      $("fixDims").value = Array.isArray(grounding.dimensions) ? grounding.dimensions.join(",") : "";
      $("fixNotes").value = "";
      $("fixOverlay").classList.add("open");
    }

    function closeFix(ev) {
      if (ev && ev.target && ev.target.id !== "fixOverlay") return;
      $("fixOverlay").classList.remove("open");
    }

    async function submitFix() {
      if (!state.fixContext) return;
      const issue = $("fixIssue").value.trim();
      const keyword = $("fixKeyword").value.trim();
      const domain = $("fixDomain").value.trim() || "general";
      const table = $("fixTable").value.trim();
      const metric = $("fixMetric").value.trim();
      const dims = $("fixDims").value.split(",").map(v => v.trim()).filter(Boolean);
      const notes = $("fixNotes").value.trim();
      if (!issue || !keyword || !table || !metric) {
        toast("Issue, keyword, table and metric are required.");
        return;
      }
      try {
        await apiClient.submitFix({
          db_connection_id: state.connectionId || "default",
          trace_id: state.fixContext.trace_id,
          session_id: state.fixContext.session_id,
          goal: state.fixContext.goal,
          issue,
          keyword,
          domain,
          target_table: table,
          target_metric: metric,
          target_dimensions: dims,
          notes,
        });
        closeFix();
        toast("Fix applied. Re-running query with updated rule...");
        const previousGoal = state.fixContext.goal;
        state.fixContext = null;
        $("goalInput").value = previousGoal;
        await runQuery();
        await refreshRules();
      } catch (err) {
        toast((err && err.message) ? err.message : "Fix failed.");
      }
    }

    function openRules() {
      $("rulesOverlay").classList.add("open");
      resetRuleForm();
      refreshRules();
    }

    function closeRules(ev) {
      if (ev && ev.target && ev.target.id !== "rulesOverlay") return;
      $("rulesOverlay").classList.remove("open");
    }

    function resetRuleForm() {
      $("ruleEditId").value = "";
      $("ruleFormTitle").textContent = "Create Rule";
      $("ruleSaveBtn").textContent = "Create Rule";
      $("ruleCancelBtn").style.display = "none";
      $("ruleName").value = "";
      $("ruleDomain").value = "general";
      $("ruleType").value = "plan_override";
      $("ruleStatus").value = "active";
      $("ruleTriggers").value = "";
      $("rulePayload").value = "";
      $("ruleNotes").value = "";
    }

    function startRuleEditByIndex(idx) {
      const rule = state.ruleCache[idx];
      if (!rule) {
        toast("Rule not found.");
        return;
      }
      $("ruleEditId").value = String(rule.rule_id || "");
      $("ruleFormTitle").textContent = `Edit Rule • ${String(rule.rule_id || "").slice(0, 12)}`;
      $("ruleSaveBtn").textContent = "Update Rule";
      $("ruleCancelBtn").style.display = "inline-flex";
      $("ruleName").value = String(rule.name || "");
      $("ruleDomain").value = String(rule.domain || "general");
      $("ruleType").value = String(rule.rule_type || "plan_override");
      $("ruleStatus").value = String(rule.status || "active");
      $("ruleTriggers").value = Array.isArray(rule.triggers) ? rule.triggers.join(", ") : "";
      $("rulePayload").value = JSON.stringify(rule.action_payload || {}, null, 2);
      $("ruleNotes").value = String(rule.notes || "");
    }

    async function saveRule() {
      const name = $("ruleName").value.trim();
      const domain = $("ruleDomain").value.trim() || "general";
      const ruleType = $("ruleType").value;
      const status = $("ruleStatus").value;
      const triggers = $("ruleTriggers").value.split(",").map(v => v.trim()).filter(Boolean);
      const notes = $("ruleNotes").value.trim();
      const editId = $("ruleEditId").value.trim();
      if (!name || !triggers.length) {
        toast("Rule name and at least one trigger are required.");
        return;
      }
      let payload = {};
      const payloadText = $("rulePayload").value.trim();
      if (payloadText) {
        try { payload = JSON.parse(payloadText); }
        catch { toast("Invalid JSON in action payload."); return; }
      }
      try {
        const isEdit = Boolean(editId);
        const path = isEdit ? "/api/assistant/rules/update" : "/api/assistant/rules";
        const body = {
          db_connection_id: state.connectionId || "default",
          domain,
          name,
          rule_type: ruleType,
          triggers,
          action_payload: payload,
          notes,
          priority: 1.0,
          status
        };
        if (isEdit) {
          body.rule_id = editId;
          body.note = "admin_ui_edit";
        }
        await apiClient.upsertRule(path, body);
        toast(isEdit ? "Rule updated." : "Rule created.");
        resetRuleForm();
        await refreshRules();
      } catch (err) {
        toast((err && err.message) ? err.message : "Rule save failed.");
      }
    }

    function getRuleIdByIndex(idx) {
      const rule = state.ruleCache[idx];
      return rule ? String(rule.rule_id || "") : "";
    }

    async function updateRuleStatusByIndex(idx, status) {
      const ruleId = getRuleIdByIndex(idx);
      if (!ruleId) {
        toast("Rule not found.");
        return;
      }
      try {
        await apiClient.setRuleStatus({
          db_connection_id: state.connectionId || "default",
          rule_id: ruleId,
          status,
          note: "admin_ui_status_change",
        });
        toast(`Rule ${status}.`);
        await refreshRules();
      } catch (err) {
        toast((err && err.message) ? err.message : "Status update failed.");
      }
    }

    async function rollbackRuleByIndex(idx) {
      const ruleId = getRuleIdByIndex(idx);
      if (!ruleId) {
        toast("Rule not found.");
        return;
      }
      try {
        await apiClient.rollbackRule({
          db_connection_id: state.connectionId || "default",
          rule_id: ruleId,
        });
        toast("Rule rolled back.");
        await refreshRules();
      } catch (err) {
        toast((err && err.message) ? err.message : "Rollback failed.");
      }
    }

    async function refreshRules(retry=true) {
      const container = $("rulesList");
      container.innerHTML = `<div class="subtle">Loading rules...</div>`;
      try {
        const data = await apiClient.listRules(state.connectionId || "default", 300);
        const rules = Array.isArray(data.rules) ? data.rules : [];
        state.ruleCache = rules;
        if (!rules.length) {
          container.innerHTML = `<div class="subtle">No rules yet.</div>`;
          return;
        }
        container.innerHTML = rules.map((rule, idx) => {
          const status = String(rule.status || "").toLowerCase();
          const statusChip = `<span class="chip ${status==="active"?"good":"warn"}">${esc(status)}</span>`;
          return `
            <div class="rule-row">
              <div class="rule-title">
                <span>${esc(rule.name || rule.rule_id)}</span>
                ${statusChip}
              </div>
              <div class="rule-meta">
                <span>${esc(rule.domain || "general")}</span>
                <span>type=${esc(rule.rule_type || "")}</span>
                <span>v${fmt(rule.version)}</span>
                <span>triggers=${(rule.triggers || []).map(t => esc(t)).join(" | ")}</span>
              </div>
              <details><summary>action payload</summary><pre>${esc(JSON.stringify(rule.action_payload || {}, null, 2))}</pre></details>
              <div class="rule-actions">
                <button class="btn small" onclick="startRuleEditByIndex(${idx})">Edit</button>
                <button class="btn small" onclick="updateRuleStatusByIndex(${idx},'active')">Activate</button>
                <button class="btn small" onclick="updateRuleStatusByIndex(${idx},'disabled')">Disable</button>
                <button class="btn small" onclick="updateRuleStatusByIndex(${idx},'archived')">Archive</button>
                <button class="btn small" onclick="rollbackRuleByIndex(${idx})">Rollback</button>
              </div>
            </div>
          `;
        }).join("");
      } catch (err) {
        const msg = (err && err.message) ? String(err.message) : "Failed to load rules.";
        if (retry && /Database not found for connection/i.test(msg) && state.connectionId !== "default") {
          state.connectionId = "default";
          localStorage.setItem(STORAGE_CONN, state.connectionId);
          saveWorkspace();
          await loadConnections();
          toast("Previous connection is unavailable. Switched to default.");
          await refreshRules(false);
          return;
        }
        container.innerHTML = `<div class="subtle" style="color:#ff95a0">${esc(msg)}</div>`;
      }
    }

    async function loadConnections() {
      const sel = $("connSelect");
      try {
        const data = await apiClient.listConnections();
        const items = Array.isArray(data.connections) ? data.connections : [];
        sel.innerHTML = items.map(c => `<option value="${esc(c.id)}">${esc(c.id)} (${esc(c.kind)})${c.is_default ? " • default" : ""}</option>`).join("");
        if (!items.length) sel.innerHTML = `<option value="default">default</option>`;
        const validIds = new Set(items.map(c => String(c.id || "")));
        if (!validIds.has(state.connectionId)) {
          const preferred = items.find(c => c && c.exists) || items.find(c => c && c.is_default) || items[0];
          state.connectionId = preferred ? String(preferred.id || "default") : "default";
          localStorage.setItem(STORAGE_CONN, state.connectionId);
          saveWorkspace();
        }
        if (state.connectionId) sel.value = state.connectionId;
        sel.onchange = () => {
          state.connectionId = sel.value;
          localStorage.setItem(STORAGE_CONN, state.connectionId);
          saveWorkspace();
        };
      } catch {
        sel.innerHTML = `<option value="default">default</option>`;
      }
    }

    function _pickDefaultModel(provider, options) {
      if (!Array.isArray(options) || !options.length) return "";
      if (provider === "local") {
        const installed = options.filter(o => o && o.installed);
        const preferred = installed.find(o => /qwen2\.5:7b-instruct/i.test(String(o.name || "")));
        return (preferred || installed[0] || options[0]).name || "";
      }
      const soft = options.find(o => /mini|haiku/i.test(String(o.name || "")));
      return (soft || options[0]).name || "";
    }

    function _providerForMode(mode) {
      if (mode === "local") return "local";
      if (mode === "openai") return "openai";
      if (mode === "anthropic") return "anthropic";
      return "";
    }

    function renderModelSelector() {
      const mode = state.mode || "auto";
      const modelSelect = $("modelSelect");
      const pullBtn = $("pullModelBtn");
      const provider = _providerForMode(mode);

      if (!provider) {
        modelSelect.innerHTML = `<option value="">auto-select provider/model</option>`;
        modelSelect.disabled = true;
        pullBtn.style.display = "none";
        return;
      }

      const catalog = state.modelCatalog[provider] || {available: false, options: [], reason: "unavailable"};
      const options = Array.isArray(catalog.options) ? catalog.options : [];
      if (!options.length) {
        modelSelect.innerHTML = `<option value="">${esc(catalog.reason || "no models available")}</option>`;
        modelSelect.disabled = true;
        pullBtn.style.display = "none";
        return;
      }

      modelSelect.disabled = false;
      const selected = state.modelSelections[provider] || _pickDefaultModel(provider, options);
      if (!state.modelSelections[provider] && selected) {
        state.modelSelections[provider] = selected;
      }

      modelSelect.innerHTML = options.map(opt => {
        const label = provider === "local"
          ? `${opt.name}${opt.installed ? " • installed" : " • download"}`
          : `${opt.name} • ${opt.tier || "balanced"}`;
        const installedAttr = provider === "local" ? `data-installed="${opt.installed ? "1" : "0"}"` : "";
        return `<option value="${esc(opt.name)}" ${installedAttr}>${esc(label)}</option>`;
      }).join("");
      modelSelect.value = state.modelSelections[provider] || options[0].name;
      state.modelSelections[provider] = modelSelect.value;
      saveWorkspace();

      modelSelect.onchange = () => {
        state.modelSelections[provider] = modelSelect.value;
        saveWorkspace();
        renderModelSelector();
      };

      if (provider === "local") {
        const chosen = options.find(o => o.name === modelSelect.value);
        const needsDownload = chosen && !chosen.installed;
        pullBtn.style.display = needsDownload ? "inline-flex" : "none";
      } else {
        pullBtn.style.display = "none";
      }
    }

    async function refreshModelCatalogs() {
      try {
        const [localData, openaiData, anthropicData] = await Promise.all([
          apiClient.modelCatalog("local").catch(() => ({})),
          apiClient.modelCatalog("openai").catch(() => ({})),
          apiClient.modelCatalog("anthropic").catch(() => ({})),
        ]);
        if (localData && typeof localData === "object") state.modelCatalog.local = localData;
        if (openaiData && typeof openaiData === "object") state.modelCatalog.openai = openaiData;
        if (anthropicData && typeof anthropicData === "object") state.modelCatalog.anthropic = anthropicData;
      } catch {
        // keep previous state; user can refresh providers/models manually.
      }
      renderModelSelector();
    }

    async function pullSelectedLocalModel() {
      const model = ($("modelSelect").value || "").trim();
      if (!model) {
        toast("Choose a local model first.");
        return;
      }
      const btn = $("pullModelBtn");
      btn.disabled = true;
      try {
        await apiClient.pullLocalModel({model, activate_after_download: true});
        toast(`Downloaded ${model}.`);
        await refreshModelCatalogs();
      } catch (err) {
        toast((err && err.message) ? err.message : "Download failed.");
      } finally {
        btn.disabled = false;
      }
    }

    async function refreshProviders() {
      try {
        const data = await apiClient.providers();
        state.providers = data.checks || {};
        const status = $("providerStatus");
        if (status) {
          const checks = state.providers || {};
          const rows = [
            ["ollama", "Local (Ollama)"],
            ["openai", "Cloud (OpenAI)"],
            ["anthropic", "Cloud (Anthropic)"],
          ];
          status.innerHTML = rows.map(([key, label]) => {
            const item = checks[key] || {};
            const available = !!item.available;
            const dotClass = available ? "provider-dot available" : "provider-dot unavailable";
            const reason = available ? "ready" : (item.reason || "unavailable");
            return `<div class="provider-row"><span class="${dotClass}"></span><span class="provider-name">${esc(label)}</span><span class="provider-reason">${esc(reason)}</span></div>`;
          }).join("");
        }
        await refreshModelCatalogs();
        toast("Provider status refreshed.");
      } catch {
        toast("Failed to refresh provider status.");
      }
    }

    async function loadProviders() {
      return refreshProviders();
    }

    async function checkHealth() {
      try {
        const data = await apiClient.health();
        $("healthLabel").textContent = `${data.status || "unknown"} • ${data.semantic_ready ? "semantic ready" : "semantic loading"}`;
        $("healthDot").classList.toggle("bad", !(data.status === "ok" || data.semantic_ready));
      } catch {
        $("healthLabel").textContent = "offline";
        $("healthDot").classList.add("bad");
      }
    }

    function toast(msg) {
      const el = $("toast");
      el.textContent = msg;
      el.classList.add("show");
      setTimeout(() => el.classList.remove("show"), 2400);
    }

    function initMode() {
      const mode = $("modeSelect");
      mode.value = state.mode;
      mode.onchange = () => {
        state.mode = mode.value;
        saveWorkspace();
        renderModelSelector();
      };
    }

    function initInput() {
      $("goalInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          runQuery();
        }
      });
    }

    function init() {
      initMode();
      initInput();
      renderAll();
      loadConnections();
      loadProviders();
      refreshModelCatalogs();
      checkHealth();
      setInterval(checkHealth, 12000);
    }

    init();
