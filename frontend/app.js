const state = {
  mode: "study",
  source: "realtime",
  morningFeedback: null,
  timer: null,
};

const el = (id) => document.getElementById(id);

function setText(id, value) {
  const node = el(id);
  if (!node) return;
  node.textContent = value ?? "N/A";
}

function sanitizeLabels(isoList) {
  if (!Array.isArray(isoList)) return [];
  return isoList.map((t) => {
    if (!t) return "";
    try {
      // Keep it short for charts.
      const d = new Date(t);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return String(t).slice(0, 19);
    }
  });
}

let charts = null;
function initCharts() {
  if (charts) return;

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: { ticks: { maxTicksLimit: 6 } },
      y: { beginAtZero: false },
    },
  };

  const make = (canvasId, color, label) => {
    const ctx = el(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label,
            data: [],
            borderColor: color,
            backgroundColor: color,
            pointRadius: 1.2,
            pointHoverRadius: 3.0,
            pointBorderWidth: 0,
            tension: 0.15,
          },
        ],
      },
      options: chartOpts,
    });
  };

  charts = {
    temp: make("chart-temp", "#2563eb", "Temperature"),
    humidity: make("chart-humidity", "#16a34a", "Humidity"),
    eco2: make("chart-eco2", "#dc2626", "eCO2"),
    tvoc: make("chart-tvoc", "#7c3aed", "TVOC"),
  };
}

function updateCharts(trends) {
  if (!charts) return;
  const labels = sanitizeLabels(trends?.timestamps);
  const offline = state.source === "offline";
  const pointRadius = offline ? 2.2 : 1.2;
  const pointHoverRadius = offline ? 3.6 : 3.0;
  const tension = offline ? 0.06 : 0.15;

  charts.temp.data.labels = labels;
  charts.temp.data.datasets[0].data = trends?.temp_C ?? [];
  charts.temp.data.datasets[0].pointRadius = pointRadius;
  charts.temp.data.datasets[0].pointHoverRadius = pointHoverRadius;
  charts.temp.data.datasets[0].tension = tension;
  charts.temp.update();

  charts.humidity.data.labels = labels;
  charts.humidity.data.datasets[0].data = trends?.humidity ?? [];
  charts.humidity.data.datasets[0].pointRadius = pointRadius;
  charts.humidity.data.datasets[0].pointHoverRadius = pointHoverRadius;
  charts.humidity.data.datasets[0].tension = tension;
  charts.humidity.update();

  charts.eco2.data.labels = labels;
  charts.eco2.data.datasets[0].data = trends?.eco2_ppm ?? [];
  charts.eco2.data.datasets[0].pointRadius = pointRadius;
  charts.eco2.data.datasets[0].pointHoverRadius = pointHoverRadius;
  charts.eco2.data.datasets[0].tension = tension;
  charts.eco2.update();

  charts.tvoc.data.labels = labels;
  charts.tvoc.data.datasets[0].data = trends?.tvoc ?? [];
  charts.tvoc.data.datasets[0].pointRadius = pointRadius;
  charts.tvoc.data.datasets[0].pointHoverRadius = pointHoverRadius;
  charts.tvoc.data.datasets[0].tension = tension;
  charts.tvoc.update();
}

function updateUIForMode() {
  const isSleep = state.mode === "sleep";
  el("sleep-checkin").style.display = isSleep ? "block" : "none";
  el("btn-study")?.classList.toggle("active", state.mode === "study");
  el("btn-sleep")?.classList.toggle("active", state.mode === "sleep");
  el("btn-checkin-slept")?.classList.toggle("active", state.morningFeedback === "slept_well");
  el("btn-checkin-okay")?.classList.toggle("active", state.morningFeedback === "okay");
  el("btn-checkin-poor")?.classList.toggle("active", state.morningFeedback === "poor_sleep");

  if (state.mode === "study") {
    el("out-title-1").textContent = "Current room state";
    el("out-title-2").textContent = "Best next action";
    el("trends-subtitle").textContent = "Past 5 minutes";
    el("out-subtitle-1").textContent = "";
    el("out-subtitle-2").textContent = "";
  } else {
    el("out-title-1").textContent = "Sleep readiness";
    el("out-title-2").textContent = "Best bedtime action";
    el("trends-subtitle").textContent = "Past 30 minutes";
    el("out-subtitle-1").textContent = "";
    el("out-subtitle-2").textContent = "";
  }
}

function startAutoRefresh() {
  if (state.timer) clearInterval(state.timer);
  const ms = state.source === "realtime" ? 10_000 : 60_000;
  state.timer = setInterval(refreshView, ms);
}

async function refreshView() {
  try {
    const url = `/api/view?mode=${encodeURIComponent(state.mode)}&source=${encodeURIComponent(state.source)}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${text}`);
    }
    const data = await res.json();

    const latest = data?.latest;
    if (!latest) {
      setText("m-temp", "N/A");
      setText("m-humidity", "N/A");
      setText("m-eco2", "N/A");
      setText("m-tvoc", "N/A");
    } else {
      el("m-temp").textContent = `${Number(latest.temp_C).toFixed(1)}`;
      el("m-humidity").textContent = `${Number(latest.humidity).toFixed(0)}`;
      el("m-eco2").textContent = `${Number(latest.eco2_ppm).toFixed(0)}`;
      el("m-tvoc").textContent = `${Number(latest.tvoc).toFixed(0)}`;
    }

    setText("m-last-updated", data?.last_updated ? `Last updated: ${data.last_updated}` : "Last updated: N/A");

    const pred = data?.prediction ?? {};
    const conf = Number(pred.confidence ?? 0);
    setText("conf-value", conf.toFixed(2));
    setText("conf-label", `Label: ${pred.confidence_label ?? "Low"}`);

    if (state.mode === "study") {
      setText("out-value-1", pred.room_state ?? "N/A");
      setText("out-value-2", pred.best_action ?? "N/A");
      setText("out-explanation", pred.explanation ?? "");
    } else {
      setText("out-value-1", pred.sleep_readiness ?? "N/A");
      setText("out-value-2", pred.bedtime_action ?? "N/A");
      // For sleep mode, the "risk reason" is the closest equivalent to explanation.
      setText("out-explanation", pred.main_risk_reason ?? "");
    }

    initCharts();
    updateCharts(data?.trends ?? {});
  } catch (err) {
    console.error(err);
    setText("out-explanation", "Failed to load. Please check the backend service or CSV data.");
  }
}

async function submitMorningFeedback(morningFeedback) {
  state.morningFeedback = morningFeedback;
  updateUIForMode();
  try {
    const res = await fetch("/api/sleep/morning-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ morning_feedback: morningFeedback }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data?.detail ?? "request failed");
    }
    el("checkin-status").textContent = `Saved: ${data.morning_feedback} (${data.session_name})`;
  } catch (err) {
    console.error(err);
    el("checkin-status").textContent = `Save failed: ${err.message || err}`;
  }
}

function wireEvents() {
  el("btn-study").addEventListener("click", () => {
    state.mode = "study";
    updateUIForMode();
    refreshView();
  });

  el("btn-sleep").addEventListener("click", () => {
    state.mode = "sleep";
    updateUIForMode();
    refreshView();
  });

  el("source-realtime").addEventListener("change", () => {
    state.source = "realtime";
    startAutoRefresh();
    refreshView();
  });
  el("source-offline").addEventListener("change", () => {
    state.source = "offline";
    startAutoRefresh();
    refreshView();
  });

  el("btn-checkin-slept").addEventListener("click", () => submitMorningFeedback("slept_well"));
  el("btn-checkin-okay").addEventListener("click", () => submitMorningFeedback("okay"));
  el("btn-checkin-poor").addEventListener("click", () => submitMorningFeedback("poor_sleep"));
}

function main() {
  updateUIForMode();
  wireEvents();
  startAutoRefresh();
  refreshView();
}

main();

