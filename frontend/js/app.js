/**
 * app.js - Main Application Controller for PathSnap Evaluator Geospatial Dashboard & Teammate Benchmarks
 */

document.addEventListener("DOMContentLoaded", async () => {
  // 1. API Base URL Management (Supporting Single-Laptop & Two-Laptop LAN Setups)
  let apiBase = localStorage.getItem("pathsnap_api_url") || "";

  function getEndpoint(path) {
    if (!apiBase) return path;
    const cleanBase = apiBase.replace(/\/+$/, "");
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    return `${cleanBase}${cleanPath}`;
  }

  // 2. Initialize Leaflet Map Controller
  const mapController = new PathSnapMap("map");

  // DOM Elements - Navigation Tabs
  const navMapBtn = document.getElementById("navMapBtn");
  const navBenchmarkBtn = document.getElementById("navBenchmarkBtn");
  const mapMatchingView = document.getElementById("mapMatchingView");
  const benchmarkView = document.getElementById("benchmarkView");

  // DOM Elements - Header & Status
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const modelNameBadge = document.getElementById("modelNameBadge");
  const modelF1Badge = document.getElementById("modelF1Badge");
  const apiHostBadge = document.getElementById("apiHostBadge");

  // DOM Elements - Settings Modal
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalSaveBtn = document.getElementById("modalSaveBtn");
  const modalResetBtn = document.getElementById("modalResetBtn");
  const apiUrlInput = document.getElementById("apiUrlInput");

  // DOM Elements - Controls & Inputs
  const trajectorySelect = document.getElementById("trajectorySelect");
  const searchRadiusInput = document.getElementById("searchRadius");
  const searchRadiusVal = document.getElementById("searchRadiusVal");
  const maxCandidatesInput = document.getElementById("maxCandidates");
  const maxCandidatesVal = document.getElementById("maxCandidatesVal");
  const runBtn = document.getElementById("runBtn");
  const runBtnText = document.getElementById("runBtnText");
  const runBtnSpinner = document.getElementById("runBtnSpinner");
  const evalBtn = document.getElementById("evalBtn");
  const evalBtnText = document.getElementById("evalBtnText");
  const evalBtnSpinner = document.getElementById("evalBtnSpinner");

  // Toggle buttons
  const hmmToggle = document.getElementById("hmmToggle");
  const top1Toggle = document.getElementById("top1Toggle");
  let useHMM = true;

  // Trajectory Info Elements
  const infoTrajId = document.getElementById("infoTrajId");
  const infoSplit = document.getElementById("infoSplit");
  const infoPointCount = document.getElementById("infoPointCount");
  const infoDuration = document.getElementById("infoDuration");

  // Match Stats elements
  const statPoints = document.getElementById("statPoints");
  const statLength = document.getElementById("statLength");
  const statArcs = document.getElementById("statArcs");
  const statConnectivity = document.getElementById("statConnectivity");
  const statTime = document.getElementById("statTime");
  const statMethod = document.getElementById("statMethod");

  // Evaluation Stats elements
  const evalCard = document.getElementById("evalCard");
  const evalPlaceholder = document.getElementById("evalPlaceholder");
  const evalContent = document.getElementById("evalContent");
  const evalPointAcc = document.getElementById("evalPointAcc");
  const evalTop1Acc = document.getElementById("evalTop1Acc");
  const evalGain = document.getElementById("evalGain");
  const evalCandRecall = document.getElementById("evalCandRecall");
  const evalRoutePrec = document.getElementById("evalRoutePrec");
  const evalRouteRec = document.getElementById("evalRouteRec");
  const evalRouteF1 = document.getElementById("evalRouteF1");
  const evalJaccard = document.getElementById("evalJaccard");
  const evalGtArcs = document.getElementById("evalGtArcs");
  const evalPredArcs = document.getElementById("evalPredArcs");
  const evalTime = document.getElementById("evalTime");

  // Inspection Drawer
  const drawer = document.getElementById("bottomDrawer");
  const drawerToggle = document.getElementById("drawerToggle");
  const inspectionBody = document.getElementById("inspectionBody");

  // Layer check boxes
  const toggleGps = document.getElementById("toggleGps");
  const toggleRoute = document.getElementById("toggleRoute");
  const toggleSnapped = document.getElementById("toggleSnapped");
  const toggleGroundTruth = document.getElementById("toggleGroundTruth");
  const toggleTeammate = document.getElementById("toggleTeammate");
  const gtLayerLabel = document.getElementById("gtLayerLabel");
  const teammateLayerLabel = document.getElementById("teammateLayerLabel");

  // Benchmark Tab DOM Elements
  const kpiMacroAcc = document.getElementById("kpiMacroAcc");
  const kpiMacroF1 = document.getElementById("kpiMacroF1");
  const kpiMacroConn = document.getElementById("kpiMacroConn");
  const kpiImplCount = document.getElementById("kpiImplCount");
  const kpiBestImpl = document.getElementById("kpiBestImpl");
  const kpiAvgTime = document.getElementById("kpiAvgTime");
  const benchTrajFilter = document.getElementById("benchTrajFilter");
  const benchImplFilter = document.getElementById("benchImplFilter");
  const benchmarkTableBody = document.getElementById("benchmarkTableBody");
  const tableStatusText = document.getElementById("tableStatusText");
  const refreshBenchmarkBtn = document.getElementById("refreshBenchmarkBtn");

  // Teammate Import Modal Elements
  const importTeammateBtn = document.getElementById("importTeammateBtn");
  const importModal = document.getElementById("importModal");
  const importModalCloseBtn = document.getElementById("importModalCloseBtn");
  const importCancelBtn = document.getElementById("importCancelBtn");
  const importSubmitBtn = document.getElementById("importSubmitBtn");
  const tabUploadFile = document.getElementById("tabUploadFile");
  const tabPasteJson = document.getElementById("tabPasteJson");
  const fileUploadContainer = document.getElementById("fileUploadContainer");
  const pasteJsonContainer = document.getElementById("pasteJsonContainer");
  const dropZone = document.getElementById("dropZone");
  const browseFileBtn = document.getElementById("browseFileBtn");
  const teammateFileInput = document.getElementById("teammateFileInput");
  const selectedFileName = document.getElementById("selectedFileName");
  const jsonTextarea = document.getElementById("jsonTextarea");
  const importFeedback = document.getElementById("importFeedback");

  // Internal State Caches
  let lastMatchResult = null;
  let activeMatchedPoints = [];
  let comparisonDataCache = null;
  let currentSortColumn = "trajectory_id";
  let currentSortAsc = true;

  // Chart instances
  let chartF1 = null;
  let chartAcc = null;
  let chartRadar = null;
  let chartLat = null;

  // ==============================================================
  // VIEW NAVIGATION & TAB SWITCHING
  // ==============================================================
  navMapBtn.addEventListener("click", () => {
    navMapBtn.classList.add("active");
    navBenchmarkBtn.classList.remove("active");
    mapMatchingView.style.display = "flex";
    benchmarkView.style.display = "none";
    setTimeout(() => mapController.map.invalidateSize(), 50);
  });

  navBenchmarkBtn.addEventListener("click", async () => {
    navBenchmarkBtn.classList.add("active");
    navMapBtn.classList.remove("active");
    mapMatchingView.style.display = "none";
    benchmarkView.style.display = "flex";
    await loadBenchmarkDashboard();
  });

  // Update Host Badge
  function updateHostBadge() {
    if (!apiBase) {
      apiHostBadge.textContent = "API: Localhost";
    } else {
      try {
        const u = new URL(apiBase);
        apiHostBadge.textContent = `API: ${u.host}`;
      } catch (e) {
        apiHostBadge.textContent = `API: Custom`;
      }
    }
  }
  updateHostBadge();

  // Settings Modal Controls
  settingsBtn.addEventListener("click", () => {
    apiUrlInput.value = apiBase || window.location.origin;
    settingsModal.style.display = "flex";
  });

  modalCloseBtn.addEventListener("click", () => {
    settingsModal.style.display = "none";
  });

  modalSaveBtn.addEventListener("click", async () => {
    const val = apiUrlInput.value.trim().replace(/\/+$/, "");
    apiBase = val;
    localStorage.setItem("pathsnap_api_url", val);
    settingsModal.style.display = "none";
    updateHostBadge();
    await checkHealthAndModel();
    await loadTrajectories();
  });

  modalResetBtn.addEventListener("click", async () => {
    apiBase = "";
    localStorage.removeItem("pathsnap_api_url");
    apiUrlInput.value = "";
    settingsModal.style.display = "none";
    updateHostBadge();
    await checkHealthAndModel();
    await loadTrajectories();
  });

  // Slider bindings
  searchRadiusInput.addEventListener("input", (e) => {
    searchRadiusVal.textContent = `${e.target.value}m`;
  });

  maxCandidatesInput.addEventListener("input", (e) => {
    maxCandidatesVal.textContent = e.target.value;
  });

  // Algorithm Toggle
  hmmToggle.addEventListener("click", () => {
    useHMM = true;
    hmmToggle.classList.add("active");
    top1Toggle.classList.remove("active");
  });

  top1Toggle.addEventListener("click", () => {
    useHMM = false;
    top1Toggle.classList.add("active");
    hmmToggle.classList.remove("active");
  });

  // Layer Toggles
  toggleGps.addEventListener("change", (e) => mapController.toggleLayer("gps", e.target.checked));
  toggleRoute.addEventListener("change", (e) => mapController.toggleLayer("route", e.target.checked));
  toggleSnapped.addEventListener("change", (e) => mapController.toggleLayer("snapped", e.target.checked));
  toggleGroundTruth.addEventListener("change", (e) => mapController.toggleLayer("groundtruth", e.target.checked));
  if (toggleTeammate) {
    toggleTeammate.addEventListener("change", (e) => mapController.toggleLayer("teammate", e.target.checked));
  }

  // Drawer Toggle
  drawerToggle.addEventListener("click", () => {
    drawer.classList.toggle("collapsed");
  });

  // Check Health and Model Info
  async function checkHealthAndModel() {
    try {
      const res = await fetch(getEndpoint("/health"));
      if (res.ok) {
        const health = await res.json();
        statusDot.className = "badge-dot online";
        statusText.textContent = "API Online";
        modelNameBadge.textContent = health.champion_model || "Baseline XGBoost";
      } else {
        throw new Error();
      }

      const metaRes = await fetch(getEndpoint("/model-info"));
      if (metaRes.ok) {
        const meta = await metaRes.json();
        const f1 = (meta.validation_metrics["F1-Score"] * 100).toFixed(1);
        modelF1Badge.textContent = `Val F1: ${f1}%`;
      }
    } catch (err) {
      statusDot.className = "badge-dot offline";
      statusText.textContent = "Offline / Local";
      modelNameBadge.textContent = "Baseline XGBoost";
      modelF1Badge.textContent = "Val F1: 78.8%";
    }
  }

  // Load Trajectory Details
  async function loadTrajectoryDetails(tid) {
    try {
      const res = await fetch(getEndpoint(`/trajectories/${tid}`));
      if (!res.ok) return;
      const data = await res.json();
      infoTrajId.textContent = data.trajectory_id;
      infoSplit.textContent = data.split.replace("_", " ");
      infoPointCount.textContent = data.point_count.toLocaleString();
      infoDuration.textContent = `${data.duration_seconds}s`;
    } catch (e) {
      console.warn("Could not load trajectory detail:", e);
    }
  }

  // Load Trajectories Directory
  async function loadTrajectories() {
    try {
      const res = await fetch(getEndpoint("/trajectories"));
      if (!res.ok) return;
      const data = await res.json();

      trajectorySelect.innerHTML = "";

      const groups = {
        evaluator_test: document.createElement("optgroup"),
        student_test: document.createElement("optgroup"),
        val: document.createElement("optgroup"),
        train: document.createElement("optgroup"),
      };
      groups.evaluator_test.label = "Official Evaluator Dataset (10 Benchmark Trips)";
      groups.student_test.label = "Student Test Trajectories (10 Unseen Trips)";
      groups.val.label = "Validation Trajectories (Ground Truth Benchmark)";
      groups.train.label = "Training Trajectories (60 Reference Trips)";

      data.trajectories.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.trajectory_id;
        opt.textContent = `${t.trajectory_id} (${t.split.toUpperCase().replace("_", " ")})`;
        if (t.trajectory_id === "00000015") opt.selected = true;

        if (groups[t.split]) {
          groups[t.split].appendChild(opt);
        } else {
          groups.train.appendChild(opt);
        }
      });

      if (groups.evaluator_test.children.length > 0) trajectorySelect.appendChild(groups.evaluator_test);
      trajectorySelect.appendChild(groups.val);
      trajectorySelect.appendChild(groups.student_test);
      trajectorySelect.appendChild(groups.train);

      await loadTrajectoryDetails(trajectorySelect.value);
    } catch (err) {
      console.warn("Could not load trajectory list:", err);
    }
  }

  trajectorySelect.addEventListener("change", async (e) => {
    await loadTrajectoryDetails(e.target.value);
    resetEvaluationUI();
  });

  // Quick Preset Buttons
  document.querySelectorAll(".quick-btn[data-tid]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const tid = btn.getAttribute("data-tid");
      if (tid) {
        trajectorySelect.value = tid;
        await loadTrajectoryDetails(tid);
        resetEvaluationUI();
        executeMapMatch();
      }
    });
  });

  function resetEvaluationUI() {
    evalBtn.disabled = true;
    evalPlaceholder.style.display = "block";
    evalContent.style.display = "none";
    toggleGroundTruth.disabled = true;
    gtLayerLabel.style.opacity = "0.5";
    mapController.groundTruthLayer.clearLayers();
    if (toggleTeammate) {
      toggleTeammate.disabled = true;
      teammateLayerLabel.style.opacity = "0.5";
    }
    mapController.clearTeammateRoutes();
  }

  // Execute Map Matching (Step 1)
  async function executeMapMatch() {
    const tid = trajectorySelect.value;
    const radius = parseFloat(searchRadiusInput.value);
    const maxCands = parseInt(maxCandidatesInput.value, 10);

    resetEvaluationUI();
    runBtn.disabled = true;
    runBtnText.textContent = "Computing Map Match...";
    runBtnSpinner.style.display = "inline-block";

    try {
      const payload = {
        trajectory_id: tid,
        use_hmm: useHMM,
        search_radius_m: radius,
        max_candidates: maxCands,
      };

      const res = await fetch(getEndpoint("/map-match"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Map Matching Error: ${err.detail || "Server error"}`);
        return;
      }

      const result = await res.json();
      lastMatchResult = result;
      activeMatchedPoints = result.matched_points || [];

      // Render Map
      mapController.renderResults(result);

      // Populate Match Stats
      statPoints.textContent = result.total_gps_points.toLocaleString();
      const lengthKm = (result.reconstructed_route.total_length_meters / 1000).toFixed(2);
      statLength.textContent = `${lengthKm} km`;
      statArcs.textContent = result.reconstructed_route.num_arcs;
      const isConnected = result.reconstructed_route.is_fully_connected;
      statConnectivity.textContent = isConnected ? "100% Connected" : `${result.reconstructed_route.disconnected_jumps} Jumps`;
      statConnectivity.className = isConnected ? "stat-value green" : "stat-value amber";
      statTime.textContent = `${(result.processing_time_seconds * 1000).toFixed(0)} ms`;
      statMethod.textContent = useHMM ? "HMM Viterbi" : "Top-1 ML Argmax";

      // Populate Inspection Table
      populateInspectionTable(result.matched_points);

      // Enable Evaluation Button
      evalBtn.disabled = false;
      evalBtn.title = "Evaluate this map match against server-side ground truth (.route)";
    } catch (e) {
      alert(`Network or Server error: ${e.message}`);
    } finally {
      runBtn.disabled = false;
      runBtnText.textContent = "▶ Run Map Matching";
      runBtnSpinner.style.display = "none";
    }
  }

  // Execute Evaluation (Step 2 - Ground Truth Verification)
  async function executeEvaluation() {
    if (!lastMatchResult) {
      alert("Please run map matching first before requesting evaluation.");
      return;
    }

    const tid = lastMatchResult.trajectory_id;
    const matchedArcIds = lastMatchResult.matched_points.map((p) => p.matched_arc_id);
    const routeArcIds = lastMatchResult.reconstructed_route.ordered_arc_ids;

    evalBtn.disabled = true;
    evalBtnText.textContent = "Evaluating...";
    evalBtnSpinner.style.display = "inline-block";

    try {
      const evalPayload = {
        trajectory_id: tid,
        matched_arc_ids: matchedArcIds,
        reconstructed_route_arcs: routeArcIds,
      };

      const res = await fetch(getEndpoint("/evaluate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(evalPayload),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Evaluation Failed: ${err.detail || "Server error"}`);
        return;
      }

      const evalData = await res.json();

      // Display Ground Truth Route on Map
      if (evalData.ground_truth_polyline_wgs84 && evalData.ground_truth_polyline_wgs84.length > 0) {
        mapController.renderGroundTruth(evalData.ground_truth_polyline_wgs84);
        toggleGroundTruth.disabled = false;
        toggleGroundTruth.checked = true;
        gtLayerLabel.style.opacity = "1.0";
      }

      // Populate Evaluation Card
      evalPointAcc.textContent = `${evalData.point_wise_accuracy_pct}%`;
      evalTop1Acc.textContent = `${evalData.top1_baseline_accuracy_pct}%`;
      const gain = evalData.accuracy_improvement_pct;
      evalGain.textContent = `${gain >= 0 ? "+" : ""}${gain}%`;
      evalGain.className = gain >= 0 ? "stat-value green" : "stat-value amber";

      evalCandRecall.textContent = `${evalData.candidate_recall_pct}%`;
      evalRoutePrec.textContent = `${evalData.route_precision_pct}%`;
      evalRouteRec.textContent = `${evalData.route_recall_pct}%`;
      evalRouteF1.textContent = `${evalData.route_f1_score_pct}%`;
      evalJaccard.textContent = `${evalData.route_jaccard_overlap_pct}%`;

      evalGtArcs.textContent = evalData.num_ground_truth_route_arcs;
      evalPredArcs.textContent = evalData.num_predicted_route_arcs;
      evalTime.textContent = `${evalData.evaluation_time_seconds} s`;

      evalPlaceholder.style.display = "none";
      evalContent.style.display = "flex";
    } catch (e) {
      alert(`Evaluation Error: ${e.message}`);
    } finally {
      evalBtn.disabled = false;
      evalBtnText.textContent = "🎯 Re-evaluate Route";
      evalBtnSpinner.style.display = "none";
    }
  }

  // Populate Interactive Inspection Table
  function populateInspectionTable(points) {
    if (!points || points.length === 0) {
      inspectionBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 8px;">No points available</td></tr>`;
      return;
    }

    const displayPts = points.slice(0, 150);
    const rowsHtml = displayPts.map((p) => {
      const isDiff = p.matched_arc_id !== p.top1_ml_arc_id;
      const arcStyle = isDiff ? "color: var(--accent-cyan); font-weight: bold;" : "";
      return `
        <tr data-idx="${p.point_idx}">
          <td>#${p.point_idx}</td>
          <td>${p.original_latitude.toFixed(5)}, ${p.original_longitude.toFixed(5)}</td>
          <td>${p.matched_latitude.toFixed(5)}, ${p.matched_longitude.toFixed(5)}</td>
          <td style="${arcStyle}">Arc ${p.matched_arc_id} ${isDiff ? "(HMM Refined)" : ""}</td>
          <td>${p.orthogonal_distance_m.toFixed(2)} m</td>
          <td>${(p.confidence * 100).toFixed(1)}%</td>
          <td>${p.candidate_count}</td>
        </tr>
      `;
    }).join("");

    const suffix = points.length > 150 ? `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 6px;">Showing first 150 of ${points.length} points (Click any row to inspect on map)</td></tr>` : "";
    inspectionBody.innerHTML = rowsHtml + suffix;

    inspectionBody.querySelectorAll("tr[data-idx]").forEach((row) => {
      row.addEventListener("click", () => {
        inspectionBody.querySelectorAll("tr.active-row").forEach((r) => r.classList.remove("active-row"));
        row.classList.add("active-row");

        const pIdx = parseInt(row.getAttribute("data-idx"), 10);
        const pt = activeMatchedPoints.find((p) => p.point_idx === pIdx);
        if (pt) {
          mapController.highlightPoint(
            pt.original_latitude,
            pt.original_longitude,
            pt.matched_latitude,
            pt.matched_longitude,
            pt.point_idx,
            pt.matched_arc_id
          );
        }
      });
    });
  }

  runBtn.addEventListener("click", executeMapMatch);
  evalBtn.addEventListener("click", executeEvaluation);

  // File Upload Handling
  const fileUpload = document.getElementById("fileUpload");
  if (fileUpload) {
    fileUpload.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async (evt) => {
        try {
          const content = evt.target.result;
          const lines = content.trim().split("\n");
          const customPoints = [];

          lines.forEach((line) => {
            const parts = line.trim().split(/[\s,]+/);
            if (parts.length >= 3) {
              const lon = parseFloat(parts[0]);
              const lat = parseFloat(parts[1]);
              const t = parseFloat(parts[2]);
              if (!isNaN(lon) && !isNaN(lat) && !isNaN(t)) {
                customPoints.push({ longitude: lon, latitude: lat, timestamp: t });
              }
            }
          });

          if (customPoints.length < 2) {
            alert("File must contain at least 2 valid GPS points.");
            return;
          }

          resetEvaluationUI();
          runBtn.disabled = true;
          runBtnText.textContent = "Processing Custom Upload...";
          runBtnSpinner.style.display = "inline-block";

          const res = await fetch(getEndpoint("/map-match"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              points: customPoints,
              use_hmm: useHMM,
              search_radius_m: parseFloat(searchRadiusInput.value),
              max_candidates: parseInt(maxCandidatesInput.value, 10),
            }),
          });

          if (!res.ok) {
            const err = await res.json();
            alert(`Error: ${err.detail}`);
            return;
          }

          const result = await res.json();
          lastMatchResult = result;
          activeMatchedPoints = result.matched_points || [];

          mapController.renderResults(result);
          statPoints.textContent = result.total_gps_points.toLocaleString();
          statLength.textContent = `${(result.reconstructed_route.total_length_meters / 1000).toFixed(2)} km`;
          statArcs.textContent = result.reconstructed_route.num_arcs;
          statConnectivity.textContent = result.reconstructed_route.is_fully_connected ? "100% Connected" : "Has Jumps";
          statTime.textContent = `${(result.processing_time_seconds * 1000).toFixed(0)} ms`;
          populateInspectionTable(result.matched_points);
        } catch (err) {
          alert("Failed to parse file: " + err.message);
        } finally {
          runBtn.disabled = false;
          runBtnText.textContent = "▶ Run Map Matching";
          runBtnSpinner.style.display = "none";
        }
      };
      reader.readAsText(file);
    });
  }

  // ==============================================================
  // EVALUATOR BENCHMARK & COMPARISON ENGINE (PHASE 7, 8, 9, 10)
  // ==============================================================

  async function loadBenchmarkDashboard() {
    try {
      tableStatusText.textContent = "Fetching evaluator data...";

      // 1. Fetch Aggregate KPI Data
      const aggRes = await fetch(getEndpoint("/evaluator/aggregate"));
      if (aggRes.ok) {
        const agg = await aggRes.json();
        const macro = agg.macro_metrics_across_trajectories;
        kpiMacroAcc.textContent = `${macro.mean_point_wise_accuracy_pct}%`;
        kpiMacroF1.textContent = `${macro.mean_route_f1_score_pct}%`;
        kpiMacroConn.textContent = `${macro.mean_topological_connectivity_pct}%`;
        kpiAvgTime.textContent = `${macro.average_processing_time_sec} s`;
      }

      // 2. Fetch Comparison Matrix
      await fetchAndRenderComparison();

      // 3. Update Available Implementations in Filter
      const implRes = await fetch(getEndpoint("/comparison/implementations"));
      if (implRes.ok) {
        const implData = await implRes.json();
        kpiImplCount.textContent = `${implData.count} System${implData.count > 1 ? "s" : ""}`;

        // Populate dropdown
        const curVal = benchImplFilter.value;
        benchImplFilter.innerHTML = '<option value="ALL">All Available Systems</option>';
        implData.implementations.forEach((name) => {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          if (name === curVal) opt.selected = true;
          benchImplFilter.appendChild(opt);
        });
      }
    } catch (err) {
      console.error("Error loading benchmark dashboard:", err);
      tableStatusText.textContent = "Error loading benchmark metrics.";
    }
  }

  async function fetchAndRenderComparison() {
    try {
      const selectedTraj = benchTrajFilter.value;
      const selectedImpl = benchImplFilter.value;

      let url = "/comparison/results";
      const params = [];
      if (selectedTraj !== "ALL") params.push(`trajectories=${selectedTraj}`);
      if (selectedImpl !== "ALL") params.push(`implementations=${encodeURIComponent(selectedImpl)}`);
      if (params.length > 0) url += `?${params.join("&")}`;

      const res = await fetch(getEndpoint(url));
      if (!res.ok) throw new Error("Failed to load comparison results");

      const data = await res.json();
      comparisonDataCache = data;

      // Update Top KPIs
      kpiBestImpl.textContent = `Leader: ${data.best_overall_implementation} (${data.best_overall_f1}% F1)`;

      // Render Table
      renderComparisonTable(data.entries);

      // Render Charts (Phase 8)
      renderBenchmarkCharts(data.entries);

      tableStatusText.textContent = `Showing ${data.entries.length} records across ${data.implementations_compared.length} system(s)`;
    } catch (e) {
      console.warn("Comparison fetch error:", e);
    }
  }

  // Render Comparison Table (Phase 7)
  function renderComparisonTable(entries) {
    if (!entries || entries.length === 0) {
      benchmarkTableBody.innerHTML = `<tr><td colspan="11" style="text-align: center; color: var(--text-muted); padding: 16px;">No records match the current filter.</td></tr>`;
      return;
    }

    // Sort entries according to currentSortColumn
    const sorted = [...entries].sort((a, b) => {
      let va = a[currentSortColumn];
      let vb = b[currentSortColumn];
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
      if (typeof va === "string") {
        return currentSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return currentSortAsc ? va - vb : vb - va;
    });

    // Find highest metric per trajectory for green highlighting
    const bestPerTraj = {};
    entries.forEach((e) => {
      const tid = e.trajectory_id;
      if (!bestPerTraj[tid]) {
        bestPerTraj[tid] = { acc: -1, prec: -1, rec: -1, f1: -1, cand: -1, conn: -1, jacc: -1 };
      }
      const b = bestPerTraj[tid];
      if (e.accuracy > b.acc) b.acc = e.accuracy;
      if (e.precision > b.prec) b.prec = e.precision;
      if (e.recall > b.rec) b.rec = e.recall;
      if (e.f1_score > b.f1_score) b.f1 = e.f1_score;
      if (e.candidate_recall > b.cand) b.cand = e.candidate_recall;
      if (e.route_continuity > b.conn) b.conn = e.route_continuity;
      if (e.route_agreement > b.jacc) b.jacc = e.route_agreement;
    });

    const rowsHtml = sorted.map((e) => {
      const b = bestPerTraj[e.trajectory_id] || {};
      const isPathSnap = e.is_pathsnap;
      const badgeClass = isPathSnap ? "badge-pathsnap" : "badge-teammate";

      const accClass = e.accuracy >= b.acc && b.acc > 0 ? "best-value" : "";
      const precClass = e.precision != null && e.precision >= b.prec && b.prec > 0 ? "best-value" : "";
      const recClass = e.recall != null && e.recall >= b.rec && b.rec > 0 ? "best-value" : "";
      const f1Class = e.f1_score != null && e.f1_score >= b.f1 && b.f1 > 0 ? "best-value" : "";
      const candClass = e.candidate_recall != null && e.candidate_recall >= b.cand && b.cand > 0 ? "best-value" : "";
      const connClass = e.route_continuity != null && e.route_continuity >= b.conn && b.conn > 0 ? "best-value" : "";
      const jaccClass = e.route_agreement != null && e.route_agreement >= b.jacc && b.jacc > 0 ? "best-value" : "";

      return `
        <tr data-tid="${e.trajectory_id}" data-impl="${e.implementation_name}">
          <td><strong>${e.trajectory_id}</strong></td>
          <td>
            <span class="${badgeClass}">${e.implementation_name}</span>
          </td>
          <td class="${accClass}">${e.accuracy != null ? e.accuracy.toFixed(2) + "%" : "N/A"}</td>
          <td class="${precClass}">${e.precision != null ? e.precision.toFixed(2) + "%" : "N/A"}</td>
          <td class="${recClass}">${e.recall != null ? e.recall.toFixed(2) + "%" : "N/A"}</td>
          <td class="${f1Class}"><strong>${e.f1_score != null ? e.f1_score.toFixed(2) + "%" : "N/A"}</strong></td>
          <td class="${candClass}">${e.candidate_recall != null ? e.candidate_recall.toFixed(2) + "%" : "N/A"}</td>
          <td class="${connClass}">${e.route_continuity != null ? e.route_continuity.toFixed(2) + "%" : "N/A"}</td>
          <td class="${jaccClass}">${e.route_agreement != null ? e.route_agreement.toFixed(2) + "%" : "N/A"}</td>
          <td>${e.processing_time_seconds != null ? e.processing_time_seconds.toFixed(2) + "s" : "--"}</td>
          <td>
            <button class="quick-btn inspect-map-btn" data-tid="${e.trajectory_id}" style="padding: 3px 8px; font-size: 10px;">
              🗺️ Inspect on Map
            </button>
          </td>
        </tr>
      `;
    }).join("");

    benchmarkTableBody.innerHTML = rowsHtml;

    // Attach "Inspect on Map" listeners
    benchmarkTableBody.querySelectorAll(".inspect-map-btn").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const tid = btn.getAttribute("data-tid");
        await inspectTrajectoryOnMap(tid);
      });
    });
  }

  // Inspect on Map Action (Phase 9)
  async function inspectTrajectoryOnMap(tid) {
    // Switch to Map View
    navMapBtn.click();
    trajectorySelect.value = tid;
    await loadTrajectoryDetails(tid);
    resetEvaluationUI();

    try {
      const res = await fetch(getEndpoint(`/comparison/results/${tid}`));
      if (!res.ok) {
        executeMapMatch();
        return;
      }
      const data = await res.json();

      // Clear map
      mapController.clear();

      // Render Ground Truth if present
      if (data.ground_truth_polyline_wgs84 && data.ground_truth_polyline_wgs84.length > 0) {
        mapController.renderGroundTruth(data.ground_truth_polyline_wgs84);
        toggleGroundTruth.disabled = false;
        toggleGroundTruth.checked = true;
        gtLayerLabel.style.opacity = "1.0";
      }

      // Check if PathSnap route is available
      const psEntry = data.routes.find((r) => r.is_pathsnap);
      if (psEntry && psEntry.route_polyline_wgs84) {
        // Mock match result for standard map layer
        const fakeResult = {
          total_gps_points: 0,
          matched_points: [],
          reconstructed_route: {
            num_arcs: 0,
            total_length_meters: 0,
            is_fully_connected: psEntry.route_continuity > 95,
            disconnected_jumps: 0,
            route_polyline_wgs84: psEntry.route_polyline_wgs84,
          },
        };
        mapController.renderResults(fakeResult);
      } else {
        await executeMapMatch();
      }

      // Render Teammate Routes in distinct colors
      const colors = ["#f59e0b", "#06b6d4", "#ec4899", "#14b8a6", "#8b5cf6"];
      let colorIdx = 0;
      let hasTeammate = false;

      data.routes.forEach((r) => {
        if (!r.is_pathsnap && r.has_route_geometry && r.route_polyline_wgs84) {
          hasTeammate = true;
          const col = colors[colorIdx % colors.length];
          colorIdx++;
          mapController.renderTeammateRoute(r.implementation_name, col, r.route_polyline_wgs84, {
            accuracy: r.accuracy,
            f1_score: r.f1_score,
            route_continuity: r.route_continuity,
          });
        }
      });

      if (hasTeammate && toggleTeammate) {
        toggleTeammate.disabled = false;
        toggleTeammate.checked = true;
        teammateLayerLabel.style.opacity = "1.0";
      }

      mapController.fitBounds();
    } catch (e) {
      console.warn("Map inspection failed:", e);
      executeMapMatch();
    }
  }

  // Sort Table Handlers
  document.querySelectorAll("#benchmarkTable th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-sort");
      if (currentSortColumn === col) {
        currentSortAsc = !currentSortAsc;
      } else {
        currentSortColumn = col;
        currentSortAsc = false; // Descending default for metrics
      }
      if (comparisonDataCache) {
        renderComparisonTable(comparisonDataCache.entries);
      }
    });
  });

  // Filter Event Listeners
  benchTrajFilter.addEventListener("change", fetchAndRenderComparison);
  benchImplFilter.addEventListener("change", fetchAndRenderComparison);
  refreshBenchmarkBtn.addEventListener("click", loadBenchmarkDashboard);

  // Render Charts (Phase 8 Visualizations)
  function renderBenchmarkCharts(entries) {
    if (!window.Chart || !entries || entries.length === 0) return;

    // Filter to PathSnap entries for per-trajectory charts
    const psEntries = entries.filter((e) => e.is_pathsnap);
    const tids = psEntries.map((e) => e.trajectory_id);

    // 1. Chart 1: F1-Score Comparison
    const ctxF1 = document.getElementById("chartF1")?.getContext("2d");
    if (ctxF1) {
      if (chartF1) chartF1.destroy();

      // Group entries by implementation
      const implGroups = {};
      entries.forEach((e) => {
        if (!implGroups[e.implementation_name]) implGroups[e.implementation_name] = {};
        implGroups[e.implementation_name][e.trajectory_id] = e.f1_score || 0;
      });

      const colors = ["#10b981", "#f59e0b", "#06b6d4", "#a855f7"];
      let cIdx = 0;
      const datasets = Object.keys(implGroups).map((name) => {
        const col = name.includes("PathSnap") ? "#10b981" : colors[(cIdx++) % colors.length];
        return {
          label: name,
          data: tids.map((tid) => implGroups[name][tid] || 0),
          backgroundColor: col,
          borderRadius: 4,
        };
      });

      chartF1 = new Chart(ctxF1, {
        type: "bar",
        data: { labels: tids, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "#9ca3af", font: { size: 10 } } },
          },
          scales: {
            x: { grid: { color: "#1f2937" }, ticks: { color: "#9ca3af", font: { size: 10 } } },
            y: { min: 0, max: 105, grid: { color: "#1f2937" }, ticks: { color: "#9ca3af" } },
          },
        },
      });
    }

    // 2. Chart 2: Point-Wise Accuracy Comparison
    const ctxAcc = document.getElementById("chartAccuracy")?.getContext("2d");
    if (ctxAcc) {
      if (chartAcc) chartAcc.destroy();
      chartAcc = new Chart(ctxAcc, {
        type: "bar",
        data: {
          labels: tids,
          datasets: [
            {
              label: "HMM Viterbi Accuracy (%)",
              data: psEntries.map((e) => e.accuracy),
              backgroundColor: "#38bdf8",
              borderRadius: 4,
            },
            {
              label: "Candidate Recall (%)",
              data: psEntries.map((e) => e.candidate_recall || 99.9),
              backgroundColor: "#10b981",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#9ca3af", font: { size: 10 } } } },
          scales: {
            x: { grid: { color: "#1f2937" }, ticks: { color: "#9ca3af", font: { size: 10 } } },
            y: { min: 40, max: 105, grid: { color: "#1f2937" }, ticks: { color: "#9ca3af" } },
          },
        },
      });
    }

    // 3. Chart 3: Multi-Metric Radar
    const ctxRadar = document.getElementById("chartRadar")?.getContext("2d");
    if (ctxRadar) {
      if (chartRadar) chartRadar.destroy();
      chartRadar = new Chart(ctxRadar, {
        type: "line",
        data: {
          labels: tids,
          datasets: [
            {
              label: "Route Jaccard Overlap (%)",
              data: psEntries.map((e) => e.route_agreement || 0),
              borderColor: "#a855f7",
              backgroundColor: "rgba(168, 85, 247, 0.15)",
              fill: true,
              tension: 0.3,
            },
            {
              label: "Route Continuity (%)",
              data: psEntries.map((e) => e.route_continuity || 0),
              borderColor: "#f59e0b",
              backgroundColor: "transparent",
              borderDash: [4, 4],
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#9ca3af", font: { size: 10 } } } },
          scales: {
            x: { grid: { color: "#1f2937" }, ticks: { color: "#9ca3af", font: { size: 10 } } },
            y: { min: 40, max: 105, grid: { color: "#1f2937" }, ticks: { color: "#9ca3af" } },
          },
        },
      });
    }

    // 4. Chart 4: Computational Latency
    const ctxLat = document.getElementById("chartLatency")?.getContext("2d");
    if (ctxLat) {
      if (chartLat) chartLat.destroy();
      chartLat = new Chart(ctxLat, {
        type: "bar",
        data: {
          labels: tids,
          datasets: [
            {
              label: "Processing Latency (Seconds)",
              data: psEntries.map((e) => e.processing_time_seconds || 0),
              backgroundColor: "#6366f1",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#9ca3af", font: { size: 10 } } } },
          scales: {
            x: { grid: { color: "#1f2937" }, ticks: { color: "#9ca3af", font: { size: 10 } } },
            y: { grid: { color: "#1f2937" }, ticks: { color: "#9ca3af" } },
          },
        },
      });
    }
  }

  // ==============================================================
  // TEAMMATE IMPORT MODAL CONTROLS (PHASE 6 & 10)
  // ==============================================================

  importTeammateBtn.addEventListener("click", () => {
    importFeedback.style.display = "none";
    importModal.style.display = "flex";
  });

  importModalCloseBtn.addEventListener("click", () => {
    importModal.style.display = "none";
  });

  importCancelBtn.addEventListener("click", () => {
    importModal.style.display = "none";
  });

  tabUploadFile.addEventListener("click", () => {
    tabUploadFile.classList.add("active");
    tabPasteJson.classList.remove("active");
    fileUploadContainer.style.display = "block";
    pasteJsonContainer.style.display = "none";
  });

  tabPasteJson.addEventListener("click", () => {
    tabPasteJson.classList.add("active");
    tabUploadFile.classList.remove("active");
    fileUploadContainer.style.display = "none";
    pasteJsonContainer.style.display = "block";
  });

  browseFileBtn.addEventListener("click", () => {
    teammateFileInput.click();
  });

  teammateFileInput.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) {
      selectedFileName.textContent = `Selected: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
      selectedFileName.style.display = "block";
    }
  });

  // Drag and Drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      teammateFileInput.files = e.dataTransfer.files;
      const f = e.dataTransfer.files[0];
      selectedFileName.textContent = `Selected: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
      selectedFileName.style.display = "block";
    }
  });

  // Submit Import Action
  importSubmitBtn.addEventListener("click", async () => {
    importSubmitBtn.disabled = true;
    importSubmitBtn.textContent = "Validating & Importing...";
    importFeedback.style.display = "none";

    try {
      if (tabUploadFile.classList.contains("active")) {
        const file = teammateFileInput.files[0];
        if (!file) {
          alert("Please select a .csv or .json file to upload.");
          return;
        }

        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(getEndpoint("/comparison/import-file"), {
          method: "POST",
          body: formData,
        });

        const respData = await res.json();
        handleImportResponse(respData);
      } else {
        const text = jsonTextarea.value.trim();
        if (!text) {
          alert("Please paste JSON benchmark records.");
          return;
        }

        let parsed;
        try {
          parsed = JSON.parse(text);
        } catch (e) {
          showFeedback("Invalid JSON syntax: " + e.message, false);
          return;
        }

        let payload = {};
        if (Array.isArray(parsed)) {
          payload = { records: parsed };
        } else if (typeof parsed === "object") {
          payload = parsed;
        }

        const res = await fetch(getEndpoint("/comparison/import"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const respData = await res.json();
        handleImportResponse(respData);
      }
    } catch (err) {
      showFeedback(`Network Error: ${err.message}`, false);
    } finally {
      importSubmitBtn.disabled = false;
      importSubmitBtn.textContent = "Import Benchmark";
    }
  });

  function handleImportResponse(data) {
    if (data.status === "success") {
      let msg = `<strong>${data.message}</strong>`;
      if (data.validation_errors && data.validation_errors.length > 0) {
        msg += `<br><span style="color: var(--accent-amber);">Warnings: ${data.validation_errors.join("; ")}</span>`;
      }
      showFeedback(msg, true);
      // Reload benchmark after 1.5s
      setTimeout(() => {
        importModal.style.display = "none";
        loadBenchmarkDashboard();
      }, 1400);
    } else {
      const errList = data.validation_errors ? data.validation_errors.join("<br>") : (data.detail || "Validation failed.");
      showFeedback(`<strong>Import Rejected:</strong><br>${errList}`, false);
    }
  }

  function showFeedback(html, isSuccess) {
    importFeedback.style.display = "block";
    importFeedback.innerHTML = html;
    importFeedback.style.background = isSuccess ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)";
    importFeedback.style.border = isSuccess ? "1px solid var(--accent-green)" : "1px solid var(--accent-red)";
    importFeedback.style.color = isSuccess ? "#34d399" : "#f87171";
  }

  // Initial startup calls
  await checkHealthAndModel();
  await loadTrajectories();

  // Run initial match for default trajectory 00000015
  setTimeout(executeMapMatch, 500);
});
