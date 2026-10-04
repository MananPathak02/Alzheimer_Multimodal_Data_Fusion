// =========================================================
// ALZFUSION+ RESEARCH PLATFORM JAVASCRIPT
// =========================================================

// --- Centralized Model Metrics Object (Item 11h) ---
const modelMetricsData = {
    rocAuc: 0.96,
    classes: ["Normal", "MCI", "AD"],
    // 3x3 Matrix rows: True Class, columns: Predicted Class (Percentages)
    matrix: [
        [98.2, 1.8, 0.0],
        [1.5, 96.4, 2.1],
        [0.0, 1.2, 98.8]
    ]
};

// State tracking for workspace ingestion
const workspaceState = {
    mriFile: null,
    petFile: null,
    isInferenceRunning: false
};

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initWorkspaceModal();
    initDropzones();
    initCompareSlider();
    initRoadmapScroll();
    initCountUp();
    renderConfusionMatrix();
    console.log("AlzFusion+ frontend initialized successfully.");
});

// =========================================================
// 1. NAVIGATION, SCROLL PROGRESS & ACTIVE LINK OBSERVER
// =========================================================
function initNavigation() {
    const navbar = document.querySelector(".navbar");
    const menuButton = document.querySelector(".menu-button");
    const navLinks = document.querySelector(".nav-links");
    const progressBar = document.querySelector(".navbar-scroll-progress");

    // Mobile menu toggle (<= 900px)
    if (menuButton && navbar) {
        menuButton.addEventListener("click", () => {
            navbar.classList.toggle("menu-open");
        });
    }

    // Close menu when clicking navigation link
    if (navLinks && navbar) {
        navLinks.querySelectorAll("a").forEach(link => {
            link.addEventListener("click", () => {
                navbar.classList.remove("menu-open");
            });
        });
    }

    // Navbar 2px scroll progress bar (Item 10e)
    window.addEventListener("scroll", () => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, scrollPercent))}%`;
        }
    }, { passive: true });

    // Active nav link highlight using IntersectionObserver
    const navAnchors = document.querySelectorAll(".nav-links a[href^='#']");
    const sectionIds = ["research", "methodology", "analysis"];
    const sections = sectionIds.map(id => document.getElementById(id)).filter(Boolean);

    if (sections.length > 0 && "IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute("id");
                    navAnchors.forEach(a => {
                        const href = a.getAttribute("href").replace("#", "");
                        if (href === id) {
                            a.style.background = "#ffffff";
                            a.style.borderColor = "var(--blue)";
                            a.style.color = "var(--blue)";
                        } else {
                            a.style.background = "";
                            a.style.borderColor = "";
                            a.style.color = "";
                        }
                    });
                }
            });
        }, {
            rootMargin: "-20% 0px -60% 0px",
            threshold: 0.1
        });

        sections.forEach(s => observer.observe(s));
    }
}

// =========================================================
// 2. INTERACTIVE WORKSPACE MODAL
// =========================================================
function openWorkspaceModal() {
    const modal = document.getElementById("workspaceModal");
    if (modal) {
        modal.classList.add("active");
        document.body.style.overflow = "hidden"; // Lock background scroll
    }
}

function closeWorkspaceModal() {
    const modal = document.getElementById("workspaceModal");
    if (modal) {
        modal.classList.remove("active");
        document.body.style.overflow = ""; // Restore background scroll
    }
}

function initWorkspaceModal() {
    const modal = document.getElementById("workspaceModal");
    if (!modal) return;

    // Close on overlay click
    modal.addEventListener("click", (e) => {
        if (e.target === modal) {
            closeWorkspaceModal();
        }
    });

    // Close on Escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.classList.contains("active")) {
            closeWorkspaceModal();
        }
    });
}

// =========================================================
// 3. DROPZONE INGESTION & FILE VALIDATION
// =========================================================
function isValidMedicalScan(file) {
    if (!file) return false;
    const name = file.name.toLowerCase();
    return name.endsWith(".nii") || name.endsWith(".nii.gz") || name.endsWith(".dcm");
}

function initDropzones() {
    const mriDropzone = document.getElementById("dropzoneMri");
    const petDropzone = document.getElementById("dropzonePet");
    const mriInput = document.getElementById("mriFileInput");
    const petInput = document.getElementById("petFileInput");
    const errorBox = document.getElementById("dropzoneError");

    function setupDropzone(box, input, stateKey, nameElemId) {
        if (!box || !input) return;

        // Click on box triggers file picker
        box.addEventListener("click", (e) => {
            if (e.target !== input) {
                input.click();
            }
        });

        // Drag & drop visual feedback
        ["dragenter", "dragover"].forEach(eventName => {
            box.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                box.classList.add("drag-over");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            box.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                box.classList.remove("drag-over");
            }, false);
        });

        // Drop file
        box.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                handleFileSelect(files[0], box, stateKey, nameElemId);
            }
        });

        // Input change
        input.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0], box, stateKey, nameElemId);
            }
        });
    }

    function handleFileSelect(file, box, stateKey, nameElemId) {
        if (!isValidMedicalScan(file)) {
            if (errorBox) {
                errorBox.innerText = `Invalid file "${file.name}". Accepted formats: .nii, .nii.gz, .dcm only.`;
                errorBox.style.display = "block";
            }
            return;
        }

        // Valid file
        if (errorBox) errorBox.style.display = "none";
        workspaceState[stateKey] = file;

        box.classList.add("has-file");
        const nameIndicator = document.getElementById(nameElemId);
        if (nameIndicator) {
            nameIndicator.innerText = file.name;
        }
    }

    setupDropzone(mriDropzone, mriInput, "mriFile", "mriFileName");
    setupDropzone(petDropzone, petInput, "petFile", "petFileName");
}

// =========================================================
// 4. SIMULATE INFERENCE & RESULT PANEL ANIMATION (Item 10b)
// =========================================================
function simulateInference() {
    const errorBox = document.getElementById("dropzoneError");
    const runBtn = document.getElementById("runInferenceBtn");
    const progressArea = document.getElementById("modalProgressArea");
    const progressText = document.getElementById("progressStepText");
    const progressBar = document.getElementById("modalProgressBar");
    const ingestionView = document.getElementById("modalIngestionView");
    const resultView = document.getElementById("modalResultView");

    // Validation: Require both paired scans
    if (!workspaceState.mriFile || !workspaceState.petFile) {
        if (errorBox) {
            errorBox.innerText = "Please provide both paired scans (Structural MRI and Metabolic PET) to proceed.";
            errorBox.style.display = "block";
        }
        return;
    }

    if (workspaceState.isInferenceRunning) return;
    workspaceState.isInferenceRunning = true;

    if (errorBox) errorBox.style.display = "none";
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.style.opacity = "0.7";
        runBtn.innerText = "Analyzing Multimodal Volumes...";
    }
    if (progressArea) progressArea.style.display = "block";

    // Step 1: Validating DICOM
    if (progressText) progressText.innerText = "Step 1/3: Validating spatial matrices & DICOM tags...";
    if (progressBar) progressBar.style.width = "30%";

    setTimeout(() => {
        // Step 2: Coordinate Space Alignment
        if (progressText) progressText.innerText = "Step 2/3: Aligning structural & metabolic coordinate spaces...";
        if (progressBar) progressBar.style.width = "65%";

        setTimeout(() => {
            // Step 3: Transformer Tensor Cross-Attention
            if (progressText) progressText.innerText = "Step 3/3: Executing Cross-Attention Tensor Fusion...";
            if (progressBar) progressBar.style.width = "100%";

            setTimeout(() => {
                // Analysis Complete: Reveal result card
                workspaceState.isInferenceRunning = false;
                if (ingestionView) ingestionView.style.display = "none";
                if (resultView) resultView.style.display = "block";

                // Animate horizontal probability bars (Normal, MCI, AD)
                animateResultBars();

                // TODO: Connect to backend Flask service
                // Example:
                // const formData = new FormData();
                // formData.append('mri', workspaceState.mriFile);
                // formData.append('pet', workspaceState.petFile);
                // fetch('/api/v1/infer', { method: 'POST', body: formData })
                //   .then(res => res.json())
                //   .then(data => updateWithRealInference(data));

            }, 700);
        }, 800);
    }, 800);
}

function animateResultBars() {
    const barNormal = document.getElementById("barNormal");
    const barMci = document.getElementById("barMci");
    const barAd = document.getElementById("barAd");

    // Target demo probabilities
    const pNormal = 0.6;
    const pMci = 98.4;
    const pAd = 1.0;

    // Reset initial widths
    if (barNormal) barNormal.style.width = "0%";
    if (barMci) barMci.style.width = "0%";
    if (barAd) barAd.style.width = "0%";

    // Trigger smooth animation
    requestAnimationFrame(() => {
        setTimeout(() => {
            if (barNormal) barNormal.style.width = `${pNormal}%`;
            if (barMci) barMci.style.width = `${pMci}%`;
            if (barAd) barAd.style.width = `${pAd}%`;
        }, 50);
    });
}

function resetWorkspaceModal() {
    workspaceState.mriFile = null;
    workspaceState.petFile = null;
    workspaceState.isInferenceRunning = false;

    // Reset dropzones
    const mriDropzone = document.getElementById("dropzoneMri");
    const petDropzone = document.getElementById("dropzonePet");
    const mriInput = document.getElementById("mriFileInput");
    const petInput = document.getElementById("petFileInput");
    const mriName = document.getElementById("mriFileName");
    const petName = document.getElementById("petFileName");
    const errorBox = document.getElementById("dropzoneError");
    const progressArea = document.getElementById("modalProgressArea");
    const progressBar = document.getElementById("modalProgressBar");
    const runBtn = document.getElementById("runInferenceBtn");
    const ingestionView = document.getElementById("modalIngestionView");
    const resultView = document.getElementById("modalResultView");

    if (mriDropzone) mriDropzone.classList.remove("has-file");
    if (petDropzone) petDropzone.classList.remove("has-file");
    if (mriInput) mriInput.value = "";
    if (petInput) petInput.value = "";
    if (mriName) mriName.innerText = "";
    if (petName) petName.innerText = "";
    if (errorBox) errorBox.style.display = "none";

    if (progressArea) progressArea.style.display = "none";
    if (progressBar) progressBar.style.width = "0%";

    if (runBtn) {
        runBtn.disabled = false;
        runBtn.style.opacity = "1";
        runBtn.innerText = "Execute Tensor Fusion Analysis";
    }

    if (resultView) resultView.style.display = "none";
    if (ingestionView) ingestionView.style.display = "block";
}

// =========================================================
// 5. COMPARE SLIDER (Item 10a)
// =========================================================
function initCompareSlider() {
    const wrapper = document.getElementById("compareWrapper");
    const overlay = document.getElementById("compareOverlay");
    const handle = document.getElementById("compareHandle");
    if (!wrapper || !overlay || !handle) return;

    let isDragging = false;

    function setSliderPosition(percent) {
        const clamped = Math.max(0, Math.min(100, percent));
        overlay.style.width = `${clamped}%`;
        handle.style.left = `${clamped}%`;
        wrapper.setAttribute("aria-valuenow", Math.round(clamped));
    }

    function handleMove(clientX) {
        const rect = wrapper.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        const percent = (offsetX / rect.width) * 100;
        setSliderPosition(percent);
    }

    // Mouse events
    wrapper.addEventListener("mousedown", (e) => {
        isDragging = true;
        handleMove(e.clientX);
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        handleMove(e.clientX);
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
    });

    // Touch events
    wrapper.addEventListener("touchstart", (e) => {
        isDragging = true;
        if (e.touches && e.touches.length > 0) {
            handleMove(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
        if (!isDragging) return;
        if (e.touches && e.touches.length > 0) {
            handleMove(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener("touchend", () => {
        isDragging = false;
    });

    // Keyboard accessibility (ArrowLeft / ArrowRight)
    wrapper.addEventListener("keydown", (e) => {
        const currentVal = parseFloat(wrapper.getAttribute("aria-valuenow")) || 50;
        if (e.key === "ArrowLeft") {
            e.preventDefault();
            setSliderPosition(currentVal - 5);
        } else if (e.key === "ArrowRight") {
            e.preventDefault();
            setSliderPosition(currentVal + 5);
        }
    });
}

// =========================================================
// 6. ROADMAP SCROLL PROGRESS & IN-VIEW DETECTOR (Item 10c)
// =========================================================
function initRoadmapScroll() {
    const roadmapContainer = document.querySelector(".ai-roadmap-container");
    const spineFill = document.getElementById("roadmapSpineFill");
    const milestones = document.querySelectorAll(".roadmap-milestone");

    if (!roadmapContainer || !spineFill) return;

    // Update spine line fill height on scroll
    window.addEventListener("scroll", () => {
        const rect = roadmapContainer.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        
        // Start filling when container enters viewport, reach 100% when last milestone is centered
        const totalHeight = rect.height;
        const progressFromTop = (viewportHeight / 2) - rect.top;
        const fillPercent = (progressFromTop / totalHeight) * 100;

        spineFill.style.height = `${Math.min(100, Math.max(0, fillPercent))}%`;
    }, { passive: true });

    // In-view milestone detector
    if ("IntersectionObserver" in window) {
        const milestoneObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("in-view");
                }
            });
        }, {
            rootMargin: "0px 0px -25% 0px",
            threshold: 0.2
        });

        milestones.forEach(m => milestoneObserver.observe(m));
    }
}

// =========================================================
// 7. COUNT-UP ANIMATION (Item 10d)
// =========================================================
function initCountUp() {
    const countElements = document.querySelectorAll("[data-count]");
    if (countElements.length === 0) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function startCountAnimation(el) {
        const target = parseFloat(el.getAttribute("data-count"));
        const prefix = el.getAttribute("data-prefix") || "";
        const suffix = el.getAttribute("data-suffix") || "";
        const isDecimal = String(target).includes(".");
        const decimalPlaces = isDecimal ? String(target).split(".")[1].length : 0;

        if (prefersReducedMotion) {
            el.innerText = `${prefix}${target.toFixed(decimalPlaces)}${suffix}`;
            return;
        }

        const duration = 1200; // 1.2s
        const startTime = performance.now();

        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(1, elapsed / duration);
            // Ease-out cubic: 1 - pow(1 - p, 3)
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const currentVal = easeOut * target;

            el.innerText = `${prefix}${currentVal.toFixed(decimalPlaces)}${suffix}`;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.innerText = `${prefix}${target.toFixed(decimalPlaces)}${suffix}`;
            }
        }

        requestAnimationFrame(update);
    }

    if ("IntersectionObserver" in window) {
        const countObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    startCountAnimation(entry.target);
                    observer.unobserve(entry.target); // Animate once
                }
            });
        }, { threshold: 0.6 });

        countElements.forEach(el => countObserver.observe(el));
    } else {
        countElements.forEach(el => startCountAnimation(el));
    }
}

// =========================================================
// 8. 3x3 CONFUSION MATRIX VISUALIZATION (Item 11h)
// =========================================================
function renderConfusionMatrix() {
    const container = document.getElementById("confusionMatrixContainer");
    if (!container || !modelMetricsData) return;

    const { classes, matrix } = modelMetricsData;
    container.innerHTML = "";

    // Top-left blank corner
    const cornerCell = document.createElement("div");
    cornerCell.className = "matrix-header-cell";
    cornerCell.innerText = "T \\ P";
    container.appendChild(cornerCell);

    // Column headers (Predicted)
    classes.forEach(cName => {
        const colHeader = document.createElement("div");
        colHeader.className = "matrix-header-cell";
        colHeader.innerText = cName;
        container.appendChild(colHeader);
    });

    // Rows (True Class)
    matrix.forEach((row, rowIdx) => {
        // Row header
        const rowHeader = document.createElement("div");
        rowHeader.className = "matrix-header-cell";
        rowHeader.innerText = classes[rowIdx];
        container.appendChild(rowHeader);

        // Cells
        row.forEach((value) => {
            const cell = document.createElement("div");
            cell.className = "matrix-cell";
            cell.innerText = `${value}%`;

            // Color intensity from tint-1 to --blue
            const intensity = Math.min(1, Math.max(0, value / 100));
            if (intensity > 0.5) {
                cell.style.background = `rgba(40, 120, 255, ${0.15 + (intensity * 0.7)})`;
                cell.style.color = intensity > 0.8 ? "#ffffff" : "var(--ink)";
            } else if (intensity > 0.05) {
                cell.style.background = `rgba(56, 189, 248, ${0.1 + (intensity * 0.4)})`;
                cell.style.color = "var(--ink)";
            } else {
                cell.style.background = "var(--tint-1)";
                cell.style.color = "#94a3b8";
            }

            container.appendChild(cell);
        });
    });
}
