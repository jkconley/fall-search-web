// app.js — runs Python in a Web Worker via Pyodide (CDN)
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

cfgFile.addEventListener("change", async () => {
  const f = cfgFile.files?.[0];
  if (!f) return;
  cfgBytes = new Uint8Array(await f.arrayBuffer());
  cfgName.textContent = `Loaded: ${f.name} (${f.size} bytes)`;
});

useCfgBtn.onclick = () => {
  if (!cfgBytes) alert("Choose a JSON file first.");
  else alert("Config will be used on next Run.");
};

function makeWorker() {
  const code = `
self.onmessage = async (e) => {
  const { n, configBytes } = e.data;
  try {
    importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");
    const pyodide = await loadPyodide();
    pyodide.setStdout({batched: (s)=> self.postMessage({type:"log", s})});
    pyodide.setStderr({batched: (s)=> self.postMessage({type:"log", s})});

    // Write your Python file
    const src = await (await fetch("fall_search_stdlib_v6_2_5_1.py")).text();
    pyodide.FS.writeFile("/fall_search.py", src);

    // Optional external config
    let useExternal = false;
    if (configBytes && configBytes.length) {
      pyodide.FS.writeFile("/ext_config.json", configBytes);
      useExternal = true;
    }

    await pyodide.runPythonAsync("import sys; sys.path.insert(0, '/'); import fall_search as fs");

    let py;
    if (useExternal) {
      // Merge external over in-file CONFIG (strict=False)
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

    const out = await pyodide.runPythonAsync(py);
    const paths = JSON.parse(out);

    // Utility to send files back
    function sendFile(p){
      try {
        const data = pyodide.FS.readFile(p);
        self.postMessage({type:"file", path:p, blob:data.buffer}, [data.buffer]);
      } catch (err) {
        // ignore missing
      }
    }

    ["csv","json","geojson","kml","spiral_geojson"].forEach(k=>{
      if (paths[k]) sendFile(paths[k]);
    });

    self.postMessage({type:"done", runDir: paths.run_dir || ""});
  } catch (err) {
    self.postMessage({type:"log", s: "\\n[ERROR] " + (err?.message || String(err)) + "\\n"});
    self.postMessage({type:"done", runDir: ""});
  }
};
  `;
  return new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
}

let worker = null;

function ensureWorker() {
  if (worker) return worker;
  worker = makeWorker();
  worker.onmessage = (e) => {
    const { type } = e.data;
    if (type === "log") {
      logEl.textContent += e.data.s;
    } else if (type === "file") {
      const name = e.data.path.split("/").pop();
      const a = document.createElement("a");
      const blob = new Blob([e.data.blob]);
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.textContent = "Download " + name;
      downloadsEl.appendChild(a);
    } else if (type === "done") {
      statusEl.textContent = "Finished.";
    }
  };
  return worker;
}

async function run(n) {
  downloadsEl.innerHTML = "";
  logEl.textContent = "";
  statusEl.textContent = "Running…";
  const w = ensureWorker();
  w.postMessage({ n, configBytes: cfgBytes });
}

btn.onclick = () => run(Number(nInput.value || 500));
btn200.onclick = () => run(200);
