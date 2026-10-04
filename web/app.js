const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const charCount = document.querySelector("#char-count");
const searchButton = document.querySelector("#search-button");
const loadingPanel = document.querySelector("#loading");
const loadingTitle = document.querySelector("#loading-title");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const resultsSection = document.querySelector("#results-section");
const resultsGrid = document.querySelector("#results-grid");
const resultCount = document.querySelector("#result-count");
const elapsedTime = document.querySelector("#elapsed-time");
const queryVariants = document.querySelector("#query-variants");
const systemState = document.querySelector("#system-state");
const stateText = document.querySelector("#state-text");
const previewDialog = document.querySelector("#preview-dialog");
const previewImage = document.querySelector("#preview-image");
const previewLocation = document.querySelector("#preview-location");
const previewCaption = document.querySelector("#preview-caption");
const closePreview = document.querySelector("#close-preview");

function setSystemState(state, vlmEnabled = true) {
  const labels = {
    cold: "Chưa nạp model",
    loading: "Đang nạp model",
    ready: vlmEnabled ? "Sẵn sàng · VLM ON" : "Sẵn sàng · VLM OFF",
    failed: "Khởi tạo thất bại",
  };
  systemState.className = `system-state ${state}`;
  stateText.textContent = labels[state] || state;
}

function setBusy(busy) {
  searchButton.disabled = busy;
  queryInput.disabled = busy;
  loadingPanel.hidden = !busy;
  if (busy) {
    loadingPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function metadataChip(label, value, className = "") {
  const chip = document.createElement("span");
  chip.className = className;
  chip.textContent = `${label} ${value ?? "—"}`;
  return chip;
}

function openPreview(result) {
  if (!result.image_url) return;
  previewImage.src = result.image_url;
  previewImage.alt = `Rank ${result.rank}, ${result.video_id}, frame ${result.frame_id}`;
  previewLocation.textContent = `${result.video_id} · IMG ${result.image_number ?? "—"} · FRAME ${result.frame_id}`;
  previewCaption.textContent = result.caption || "Không có caption.";
  previewDialog.showModal();
}

function createResultCard(result) {
  const card = document.createElement("article");
  card.className = "result-card";

  const imageButton = document.createElement("button");
  imageButton.className = "image-button";
  imageButton.type = "button";
  imageButton.setAttribute("aria-label", `Mở ảnh rank ${result.rank}`);

  if (result.image_url) {
    const image = document.createElement("img");
    image.src = result.image_url;
    image.alt = `${result.video_id}, image ${result.image_number}, frame ${result.frame_id}`;
    image.loading = result.rank <= 6 ? "eager" : "lazy";
    image.addEventListener("error", () => {
      image.remove();
      const missing = document.createElement("span");
      missing.className = "image-missing";
      missing.textContent = "IMAGE NOT FOUND";
      imageButton.prepend(missing);
    });
    imageButton.append(image);
    imageButton.addEventListener("click", () => openPreview(result));
  } else {
    const missing = document.createElement("span");
    missing.className = "image-missing";
    missing.textContent = "IMAGE NOT FOUND";
    imageButton.append(missing);
    imageButton.disabled = true;
  }

  const rank = document.createElement("span");
  rank.className = "rank-badge";
  rank.textContent = String(result.rank).padStart(2, "0");
  imageButton.append(rank);

  const content = document.createElement("div");
  content.className = "card-content";

  const video = document.createElement("strong");
  video.className = "video-id";
  video.textContent = result.video_id;

  const location = document.createElement("div");
  location.className = "location-row";
  location.append(
    metadataChip("IMG", result.image_number),
    metadataChip("FRAME", result.frame_id),
    metadataChip("SCORE", formatNumber(result.score), "score"),
  );

  const caption = document.createElement("p");
  caption.className = "caption";
  caption.textContent = result.caption || "Không có caption cho frame này.";
  caption.title = result.caption || "";

  content.append(video, location, caption);
  card.append(imageButton, content);
  return card;
}

function renderResults(payload) {
  resultsGrid.replaceChildren();
  queryVariants.replaceChildren();

  payload.query_variants.forEach((variant) => {
    const item = document.createElement("li");
    item.textContent = variant;
    queryVariants.append(item);
  });

  payload.results.forEach((result) => resultsGrid.append(createResultCard(result)));
  resultCount.textContent = `${payload.results.length} kết quả`;
  elapsedTime.textContent = `${payload.elapsed_seconds.toFixed(2)} giây`;
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function errorDetail(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg || String(item)).join("; ");
  }
  return fallback;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    setSystemState(status.state, status.vlm_enabled);
  } catch {
    setSystemState("failed");
  }
}

queryInput.addEventListener("input", () => {
  charCount.textContent = `${queryInput.value.length} / 8000`;
});

queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  errorPanel.hidden = true;
  loadingTitle.textContent = "Đang nạp model và tìm kiếm";
  setSystemState("loading");
  setBusy(true);

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    }
    renderResults(payload);
    setSystemState("ready", payload.vlm_enabled);
  } catch (error) {
    errorMessage.textContent = error.message || "Không thể hoàn thành tìm kiếm.";
    errorPanel.hidden = false;
    errorPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    await refreshStatus();
  } finally {
    setBusy(false);
  }
});

closePreview.addEventListener("click", () => previewDialog.close());
previewDialog.addEventListener("click", (event) => {
  if (event.target === previewDialog) previewDialog.close();
});

refreshStatus();
