// Base URL of the Flask backend. Change this if you deploy the API elsewhere.
const API_BASE_URL = "http://localhost:5000";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropText = document.getElementById("dropText");
const preview = document.getElementById("preview");
const classifyBtn = document.getElementById("classifyBtn");
const loading = document.getElementById("loading");
const errorMsg = document.getElementById("errorMsg");

const resultsPanel = document.getElementById("resultsPanel");
const topPrediction = document.getElementById("topPrediction");
const inferenceTime = document.getElementById("inferenceTime");
const gradcamImg = document.getElementById("gradcamImg");
const topkList = document.getElementById("topkList");

let selectedFile = null;

// --- File selection (click) ---
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
});

// --- File selection (drag & drop) ---
["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file.");
    return;
  }
  selectedFile = file;
  errorMsg.hidden = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropText.textContent = file.name;
  };
  reader.readAsDataURL(file);

  classifyBtn.disabled = false;
  resultsPanel.hidden = true;
}

// --- Classification request ---
classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  errorMsg.hidden = true;
  loading.hidden = false;
  classifyBtn.disabled = true;
  resultsPanel.hidden = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResults(data);
  } catch (err) {
    showError(err.message || "Something went wrong while contacting the API.");
  } finally {
    loading.hidden = true;
    classifyBtn.disabled = false;
  }
});

function renderResults(data) {
  topPrediction.textContent = data.predicted_class;
  inferenceTime.textContent = `Inference time: ${data.inference_time_ms} ms`;
  gradcamImg.src = data.gradcam_overlay;

  topkList.innerHTML = "";
  data.predictions.forEach((pred) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${pred.label}</span><strong>${pred.confidence}%</strong>`;
    topkList.appendChild(li);
  });

  resultsPanel.hidden = false;
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}
