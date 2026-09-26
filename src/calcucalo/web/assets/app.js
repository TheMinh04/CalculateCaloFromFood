const state = {
  file: null,
  previewUrl: null,
  result: null,
  modelReady: false,
};

const $ = (selector) => document.querySelector(selector);
const form = $("#analysisForm");
const fileInput = $("#imageInput");
const dropzone = $("#dropzone");
const analyzeButton = $("#analyzeButton");
const calibrationMode = $("#calibrationMode");
const calibrationValue = $("#calibrationValue");
const loadingOverlay = $("#loadingOverlay");
const resultsSection = $("#resultsSection");
const SVG_NS = "http://www.w3.org/2000/svg";
const BOX_COLORS = ["#ff7a45", "#c6e95d", "#5271ff", "#ffca58", "#e66bbb"];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 1) {
  const parsed = Number(value || 0);
  return new Intl.NumberFormat("vi-VN", { maximumFractionDigits: digits }).format(parsed);
}

function setMessage(message = "") {
  $("#formMessage").textContent = message;
}

function updateButton() {
  analyzeButton.disabled = !(state.file && state.modelReady);
}

async function loadRuntime() {
  const status = $("#runtimeStatus");
  const banner = $("#modelBanner");
  try {
    const healthResponse = await fetch("/health");
    const health = await healthResponse.json();
    if (!health.model_exists) {
      throw new Error("Chưa tìm thấy best.onnx/best.pt hoặc CALCUCALO_MODEL.");
    }
    const infoResponse = await fetch("/api/v1/model/info");
    const info = await infoResponse.json();
    if (!infoResponse.ok) throw new Error(info.detail || "Không đọc được metadata model.");

    state.modelReady = true;
    status.className = "runtime-status ready";
    status.querySelector("span:last-child").textContent =
      `${info.detector.model_file} · ${info.detector.class_count} lớp · ${info.detector.backend}`;

    const readiness = info.readiness || {};
    const training = info.detector.training || {};
    const metrics = info.detector.metrics || {};
    const map50 = metrics["metrics/mAP50(B)"];
    const details = [
      training.epochs ? `${training.epochs} epoch` : null,
      training.fraction ? `${number(training.fraction * 100, 0)}% dữ liệu` : null,
      map50 !== undefined ? `mAP50 ${number(map50 * 100, 2)}%` : null,
    ].filter(Boolean).join(" · ");

    banner.hidden = false;
    banner.className = `model-banner ${readiness.detector_candidate ? "ok" : ""}`;
    banner.innerHTML = `<strong>Model: ${escapeHtml(readiness.status || "unknown")}</strong>` +
      `<span>${escapeHtml(details || "Chưa có metadata train.")}</span>` +
      (readiness.warnings || []).map((warning) => `<div>• ${escapeHtml(warning)}</div>`).join("");
  } catch (error) {
    state.modelReady = false;
    status.className = "runtime-status error";
    status.querySelector("span:last-child").textContent = "Model chưa sẵn sàng";
    banner.hidden = false;
    banner.className = "model-banner";
    banner.innerHTML = `<strong>Không thể khởi tạo model</strong><span>${escapeHtml(error.message)}</span>`;
  }
  updateButton();
}

function chooseFile(file) {
  setMessage();
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setMessage("Vui lòng chọn một file ảnh JPG, PNG hoặc WebP.");
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    setMessage("Ảnh vượt quá giới hạn 15 MB.");
    return;
  }
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("#previewImage").src = state.previewUrl;
  $("#previewStage").hidden = false;
  $("#previewEmpty").hidden = true;
  $("#overlay").replaceChildren();
  $("#dropTitle").textContent = file.name;
  $("#dropHint").textContent = `${number(file.size / 1024 / 1024, 2)} MB · bấm để đổi ảnh`;
  $("#detectionCount").textContent = "Sẵn sàng phân tích";
  resultsSection.hidden = true;
  updateButton();
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

calibrationMode.addEventListener("change", () => {
  const mode = calibrationMode.value;
  calibrationValue.disabled = mode === "none";
  calibrationValue.value = "";
  if (mode === "plate") {
    $("#calibrationLabel").textContent = "Đường kính";
    $("#calibrationUnit").textContent = "cm";
    calibrationValue.placeholder = "25";
  } else if (mode === "scale") {
    $("#calibrationLabel").textContent = "Tỷ lệ";
    $("#calibrationUnit").textContent = "cm/px";
    calibrationValue.placeholder = "0.042";
  } else {
    $("#calibrationLabel").textContent = "Giá trị";
    $("#calibrationUnit").textContent = "—";
    calibrationValue.placeholder = "";
  }
});

async function analyze(overrides = null) {
  if (!state.file) return;
  const mode = calibrationMode.value;
  if (mode !== "none" && !(Number(calibrationValue.value) > 0)) {
    setMessage("Hãy nhập giá trị hiệu chuẩn lớn hơn 0.");
    return;
  }

  setMessage();
  loadingOverlay.hidden = false;
  analyzeButton.disabled = true;
  const payload = new FormData();
  payload.append("image", state.file);
  payload.append("response_format", "full");
  if (mode === "plate") payload.append("plate_diameter_cm", calibrationValue.value);
  if (mode === "scale") payload.append("cm_per_pixel", calibrationValue.value);
  if (overrides && Object.keys(overrides).length) {
    payload.append("component_overrides_json", JSON.stringify(overrides));
  }

  try {
    const response = await fetch("/api/v1/food/analyze", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `API trả lỗi ${response.status}`);
    state.result = data;
    renderResult(data);
  } catch (error) {
    setMessage(error.message || "Không thể phân tích ảnh.");
  } finally {
    loadingOverlay.hidden = true;
    updateButton();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  analyze();
});

function drawBoxes(result) {
  const overlay = $("#overlay");
  overlay.replaceChildren();
  overlay.setAttribute("viewBox", `0 0 ${result.image.width} ${result.image.height}`);
  (result.items || []).forEach((item, index) => {
    const [x1, y1, x2, y2] = item.bbox_xyxy;
    const color = BOX_COLORS[index % BOX_COLORS.length];
    const group = document.createElementNS(SVG_NS, "g");
    const rectangle = document.createElementNS(SVG_NS, "rect");
    rectangle.setAttribute("x", x1);
    rectangle.setAttribute("y", y1);
    rectangle.setAttribute("width", Math.max(1, x2 - x1));
    rectangle.setAttribute("height", Math.max(1, y2 - y1));
    rectangle.setAttribute("fill", "none");
    rectangle.setAttribute("stroke", color);
    rectangle.setAttribute("stroke-width", Math.max(2, result.image.width / 300));
    rectangle.setAttribute("vector-effect", "non-scaling-stroke");

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", x1 + 4);
    label.setAttribute("y", Math.max(16, y1 - 5));
    label.setAttribute("fill", color);
    label.setAttribute("font-size", Math.max(12, result.image.width / 42));
    label.setAttribute("font-weight", "800");
    label.setAttribute("paint-order", "stroke");
    label.setAttribute("stroke", "#102219");
    label.setAttribute("stroke-width", "3");
    label.textContent = `${item.label} ${number(item.detection_confidence * 100, 0)}%`;
    group.append(rectangle, label);
    overlay.append(group);
  });
}

function aggregateTotals(foods) {
  return foods.reduce((totals, food) => {
    const item = food.estimated_totals || {};
    totals.calories_kcal += Number(item.calories_kcal || 0);
    totals.protein_g += Number(item.protein_g || 0);
    totals.fat_g += Number(item.fat_g || 0);
    totals.carb_g += Number(item.carb_g || 0);
    return totals;
  }, { calories_kcal: 0, protein_g: 0, fat_g: 0, carb_g: 0 });
}

function renderQuality(report = {}) {
  const issueLabels = {
    low_resolution: "độ phân giải thấp",
    likely_blurry: "ảnh có thể bị mờ",
    too_dark: "ảnh quá tối",
    too_bright: "ảnh quá sáng",
  };
  const status = report.status || "unknown";
  const badge = $("#qualityBadge");
  badge.className = `quality-badge ${status}`;
  badge.textContent = status === "good"
    ? `Ảnh tốt · ${number((report.score || 0) * 100, 0)}%`
    : `Nên kiểm tra ảnh · ${number((report.score || 0) * 100, 0)}%`;

  const signals = report.signals || {};
  const issueText = (report.issues || []).length
    ? `Phát hiện: ${(report.issues || []).map((issue) => issueLabels[issue] || issue).join(", ")}.`
    : "Không phát hiện vấn đề rõ ràng về sáng, nét hoặc độ phân giải.";
  $("#qualityPanel").innerHTML = `
    <div><h3>Chất lượng đầu vào</h3><p>${escapeHtml(issueText)}</p></div>
    <div class="quality-signals">
      <span class="signal-chip">${escapeHtml(signals.width_px)} × ${escapeHtml(signals.height_px)} px</span>
      <span class="signal-chip">Độ sáng ${escapeHtml(signals.brightness_mean)}</span>
      <span class="signal-chip">Độ nét ${escapeHtml(signals.blur_variance)}</span>
    </div>`;
}

function renderFoods(foods) {
  const list = $("#foodList");
  if (!foods.length) {
    list.innerHTML = `<div class="model-banner"><strong>Chưa có món phù hợp</strong><span>Model có thể chưa nhận ra món hoặc catalog chưa có công thức tương ứng.</span></div>`;
    $("#correctionBar").hidden = true;
    return;
  }

  list.innerHTML = foods.map((food) => {
    const totals = food.estimated_totals || {};
    const components = (food.estimated_components || []).map((component) => `
      <tr>
        <td><strong>${escapeHtml(component.name)}</strong><br><span class="basis ${escapeHtml(component.basis)}">${escapeHtml(component.basis)}</span></td>
        <td><input class="component-grams" type="number" min="1" step="1"
          data-food-id="${escapeHtml(food.food_id)}"
          data-component-id="${escapeHtml(component.ingredient_id)}"
          value="${escapeHtml(component.estimated_g)}" aria-label="Gram của ${escapeHtml(component.name)}"></td>
        <td>${number(component.calories_kcal)} kcal</td>
        <td>${number(component.protein_g)} g</td>
      </tr>`).join("");
    return `
      <article class="food-card">
        <header class="food-card-header">
          <div><h3>${escapeHtml(food.name)}</h3><p>${escapeHtml(food.food_id)} · ${number(food.estimated_portion_g)} g · ${escapeHtml(food.analysis_basis)}</p></div>
          <div class="food-total"><strong>${number(totals.calories_kcal)} kcal</strong><span>Tổng ước tính</span></div>
        </header>
        <table class="component-table">
          <thead><tr><th>Thành phần / cơ sở</th><th>Khối lượng</th><th>Calories</th><th>Protein</th></tr></thead>
          <tbody>${components}</tbody>
        </table>
      </article>`;
  }).join("");
  $("#correctionBar").hidden = false;
}

function renderResult(result) {
  const foods = result.foods || [];
  const totals = aggregateTotals(foods);
  drawBoxes(result);
  $("#detectionCount").textContent = `${(result.items || []).length} vùng phát hiện`;
  $("#macroGrid").innerHTML = [
    ["Calories", totals.calories_kcal, "kcal"],
    ["Protein", totals.protein_g, "g"],
    ["Chất béo", totals.fat_g, "g"],
    ["Tinh bột", totals.carb_g, "g"],
  ].map(([label, value, unit]) => `
    <div class="macro-card"><span>${label}</span><strong>${number(value)}</strong><small>${unit}</small></div>`).join("");

  renderQuality(result.image_quality || {});
  const warnings = result.warnings || [];
  const warningPanel = $("#warningPanel");
  warningPanel.hidden = !warnings.length;
  warningPanel.innerHTML = warnings.length
    ? `<h3>Lưu ý trước khi dùng kết quả</h3><ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "";
  renderFoods(foods);
  $("#jsonOutput").textContent = JSON.stringify(result, null, 2);
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#recalculateButton").addEventListener("click", () => {
  const overrides = {};
  document.querySelectorAll(".component-grams").forEach((input) => {
    const grams = Number(input.value);
    if (!(grams > 0)) return;
    const foodId = input.dataset.foodId;
    overrides[foodId] ||= {};
    overrides[foodId][input.dataset.componentId] = grams;
  });
  analyze(overrides);
});

loadRuntime();
