// app.js — robust version with better status + error reporting + baseHref fix
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const downloadsEl = document.getElementById("downloads");
const btn = document.getElementById("run");
const btn200 = document.getElementById("demo200");
const nInput = document.getElementById("n");
const cfgFile = document.getElementById("cfgfile");
const useCfgBtn = document.getElementById("useCfg");
const cfgName = document.getElementById("cfgname");

let cfgBytes = null;
let worker = null;
let spinnerInterval = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function appendLog(line) {
  logEl.textContent += line;
}

function showError(err) {
  setStatus("Error");
  const msg = (err && err.stack) ? err.stack : String(err);
  appendLog(`\n[JS ERROR] ${msg}\n`);
  console.error(err);
}

cfgFile?.addEventListener("change", async () => {
  try {
    const f = cfgFile.files?.[0];
    if (!f) return;
    cfgBytes = new Uint8Array(await f.arrayBuffer());
    cfgName.textContent = `Loaded: ${f.name} (${f.size} bytes)`;
  } catch (e) { showError(e); }
});

useCfgBtn && (useCfgBtn.onclick = () => {
  if (!cfgBytes) alert("Choose a JSON file first.");
  else alert("Config will be used on next Run.");
});

function buildWorker() {
  const code = `
self.onmessage = async (e) => {
  const { n, configBytes, baseHref } = e.data;
  try {
    // Signal boot
    self.postMessage({ type: "log", s: "[worker] starting…\\n" });

    // Load Pyodide
    importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");
    const pyodide = await loadPyodide();
    self.postMessage({ type: "log", s: "[worker] pyodide loaded\\n" });

    // Stream Python prints back to UI
    pyodide.setStdout({ batched: (s) => self.postMessage({ type:"log", s }) });
    pyodide.setStderr({ batched: (s) => self.postMessage({ type:"log", s }) });

    // Resolve absolute URL to the Python file (fixes blob:// base issue)
    const pyUrl = new URL("fall_search_stdlib_v6_2_5_1.py", baseHref).href;
    const resp = await fetch(pyUrl);
    if (!resp.ok) throw new Error("Failed to fetch Python file: " + resp.status + " " + resp.statusText);
    const src = await resp.text();
    pyodide.FS.writeFile("/fall_search.py", src);

    // Optional external config
    let useExternal = false;
    if (configBytes && configBytes.length) {
      pyodide.FS.writeFile("/ext_config.json", configBytes);
      useExternal = true;
    }

    // Import module
    await pyodide.runPythonAsync("import sys; sys.path.insert(0, '/'); import fall_search as fs");
    self.postMessage({ type: "log", s: "[worker] fall_search imported\\n" });

    // Build Python snippet
    let py;
    if (useExternal) {
      py = \`
import json
cfg = fs.load_config_json("/ext_config.json", base=fs.CONFIG, strict=False)
cfg["N_runs"] = \${n}
res = fs.run_from_config(cfg)
json.dumps(res)
\`;
    } else {
      py = \`
import json
res = fs.run_from_config(fs.CONFIG | {"N_runs": \${n}})
json.dumps(res)
\`;
    }

    // Run simulation
    const out = await pyodide.runPythonAsync(py);
    const paths = JSON.parse(out);

    // Return files
    function sendFile(p) {
      try {
        const data = pyodide.FS.readFile(p);
        self.postMessage({ type:"file", path:p, blob:data.buffer }, [data.buffer]);
      } catch (_) { /* ignore missing */ }
    }
    ["csv","json","geojson","kml","spiral_geojson"].forEach(k => { if (paths[k]) sendFile(paths[k]); });

    self.postMessage({ type:"done", runDir: paths.run_dir || "" });
  } catch (err) {
    self.postMessage({ type:"log", s: "\\n[WORKER ERROR] " + (err?.message || String(err)) + "\\n" });
    self.postMessage({ type:"done", runDir: "" });
  }
};
`;
  return new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
}

function ensureWorker() {
  if (worker) return worker;
  try {
    worker = buildWorker();
    worker.onerror = (e) => {
      appendLog(`\n[WORKER onerror] ${e.message}\n`);
      setStatus("Error");
    };
    worker.onmessageerror = (e) => {
      appendLog(`\n[WORKER messageerror] ${e.data}\n`);
      setStatus("Error");
    };
    worker.onmessage = (e) => {
      const { type } = e.data;
      if (type === "log") {
        appendLog(e.data.s);
      } else if (type === "file") {
        const name = e.data.path.split("/").pop();
        const a = document.createElement("a");
        const blob = new Blob([e.data.blob]);
        a.href = URL.createObjectURL(blob);
        a.download = name;
        a.textContent = "Download " + name;
        downloadsEl.appendChild(a);
      } else if (type === "done") {
        clearInterval(spinnerInterval);
        btn.disabled = false;
        btn200 && (btn200.disabled = false);
        setStatus("Finished.");
      }
    };
  } catch (e) {
    showError(e);
  }
  return worker;
}

async function run(n) {
  try {
    downloadsEl.innerHTML = "";
    logEl.textContent = "";
    setStatus("Running");
    btn.disabled = true; btn200 && (btn200.disabled = true);

    // simple spinner
    spinnerInterval = setInterval(() => {
      statusEl.textContent = statusEl.textContent.endsWith("…")
        ? "Running"
        : statusEl.textContent + "…";
    }, 500);

    const w = ensureWorker();
    const BASE_HREF = new URL("./", window.location.href).href;
    w.postMessage({ n, configBytes: cfgBytes, baseHref: BASE_HREF });

    // Watchdog: if nothing logs in 10s, surface hint
    const t0 = Date.now();
    setTimeout(() => {
      if ((Date.now() - t0) > 10000 && !logEl.textContent) {
        appendLog("\n[hint] If you see nothing, press F12 and check the Console for CORS or fetch errors.\n");
      }
    }, 10050);
  } catch (e) {
    showError(e);
  }
}

btn?.addEventListener("click", () => run(Number(nInput.value || 500)));
btn200?.addEventListener("click", () => run(200));
