// Mirrors config.ALLOWED_VIDEO_EXTENSIONS / config.MAX_UPLOAD_BYTES on the
// backend — duplicated here only because the frontend has no way to read
// config.py directly; keep these in sync if those change.
const ALLOWED_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"];
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileNameEl = document.getElementById("file-name");
const analyzeButton = document.getElementById("analyze-button");
const resetButton = document.getElementById("reset-button");
const errorResetButton = document.getElementById("error-reset-button");

const uploadSection = document.getElementById("upload-section");
const processingSection = document.getElementById("processing-section");
const errorSection = document.getElementById("error-section");
const resultsSection = document.getElementById("results-section");
const errorMessageEl = document.getElementById("error-message");

let selectedFile = null;

function showSection(section) {
  for (const s of [uploadSection, processingSection, errorSection, resultsSection]) {
    s.hidden = s !== section;
  }
}

function fileExtension(name) {
  const parts = name.split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

function setSelectedFile(file) {
  if (!file) return;

  const ext = fileExtension(file.name);
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    fileNameEl.textContent = `Unsupported format ".${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(", ")}.`;
    selectedFile = null;
    analyzeButton.disabled = true;
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    fileNameEl.textContent = `File is too large (${(file.size / 1e6).toFixed(1)}MB). Max is 500MB.`;
    selectedFile = null;
    analyzeButton.disabled = true;
    return;
  }

  selectedFile = file;
  fileNameEl.textContent = `Selected: ${file.name} (${(file.size / 1e6).toFixed(1)}MB)`;
  analyzeButton.disabled = false;
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  setSelectedFile(e.dataTransfer.files[0]);
});

function resetToUpload() {
  selectedFile = null;
  fileInput.value = "";
  fileNameEl.textContent = "";
  analyzeButton.disabled = true;
  showSection(uploadSection);
}

resetButton.addEventListener("click", resetToUpload);
errorResetButton.addEventListener("click", resetToUpload);

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  showSection(processingSection);

  const formData = new FormData();
  formData.append("video", selectedFile);

  try {
    const response = await fetch("/analyze", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      errorMessageEl.textContent = data.message || "Something went wrong analyzing this video.";
      showSection(errorSection);
      return;
    }

    renderResults(data);
    showSection(resultsSection);
  } catch (err) {
    errorMessageEl.textContent = "Could not reach the server. Is it running?";
    showSection(errorSection);
  }
});

function renderResults(data) {
  const verdictLabel = document.getElementById("verdict-label");
  verdictLabel.textContent = data.prediction;
  verdictLabel.className = "verdict-label " + (data.prediction === "FAKE" ? "fake" : "real");
  document.getElementById("verdict-confidence").textContent =
    `${(data.confidence * 100).toFixed(1)}% confidence`;

  const ex = data.explainability;
  if (ex) {
    document.getElementById("img-original").src = `data:image/png;base64,${ex.original_frame}`;
    document.getElementById("img-face").src = `data:image/png;base64,${ex.face_crop}`;
    document.getElementById("img-gray").src = `data:image/png;base64,${ex.grayscale}`;
    document.getElementById("img-dft").src = `data:image/png;base64,${ex.dft_magnitude}`;
    document.getElementById("img-radial").src = `data:image/png;base64,${ex.radial_profile_plot}`;
  }

  document.getElementById("frames-summary").textContent =
    `${data.frame_results.length} frames sampled, ${data.frames_discarded} discarded ` +
    `(no usable face / too blurry), ${data.frame_results.length - data.frames_discarded} used in the vote.`;

  const tbody = document.getElementById("frames-table-body");
  tbody.innerHTML = "";
  for (const frame of data.frame_results) {
    const row = document.createElement("tr");
    if (!frame.used_in_vote) row.classList.add("discarded");

    const predCell = frame.used_in_vote
      ? `<span class="pred-${frame.prediction.toLowerCase()}">${frame.prediction}</span>`
      : "—";
    const confCell = frame.used_in_vote ? `${(frame.confidence * 100).toFixed(1)}%` : "—";

    row.innerHTML = `
      <td>${frame.frame_index}</td>
      <td>${predCell}</td>
      <td>${confCell}</td>
      <td>${frame.used_in_vote ? "yes" : "no"}</td>
    `;
    tbody.appendChild(row);
  }
}
