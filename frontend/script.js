// =========================================
// ALZFUSION FRONTEND
// =========================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("AlzFusion frontend loaded.");

});

function openWorkspaceModal() {
    document.getElementById('workspaceModal').classList.add('active');
}

function closeWorkspaceModal() {
    document.getElementById('workspaceModal').classList.remove('active');
}

function simulateInference() {
    const btn = document.querySelector('.run-inference-btn');
    btn.innerText = "Executing Cross-Attention Fusion...";
    btn.style.opacity = "0.7";
    setTimeout(() => {
        alert("Fusion analysis complete! Staging confidence: 99.2% (Early MCI Marker Detected).");
        closeWorkspaceModal();
        btn.innerText = "Execute Tensor Fusion Analysis";
        btn.style.opacity = "1";
    }, 2000);
}