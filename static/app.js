/* Frontend Interactive Controller: BIW B-Pillar Quality Dashboard */

document.addEventListener("DOMContentLoaded", () => {
    // State management variables
    let supplierData = [];
    let metaData = {};
    let scoresChart = null;
    let rejectionsChart = null;
    let featuresChart = null;
    let predictTimeout = null;

    // --- DOM Elements ---
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const tblSuppliersBody = document.querySelector("#table-suppliers tbody");
    const selSupplier = document.querySelector("#sel-supplier");
    
    // Sliders & Outputs
    const frmPredictor = document.querySelector("#frm-predictor");
    const sliders = document.querySelectorAll(".premium-slider");
    const chkboxes = document.querySelectorAll(".premium-checkbox input");
    
    // Prediction Widgets
    const lblProbability = document.querySelector("#lbl-probability");
    const lblExplanation = document.querySelector("#lbl-explanation");
    const badgeVerdict = document.querySelector("#badge-verdict");
    const gaugeFill = document.querySelector("#gauge-fill");
    const lblMargin = document.querySelector("#lbl-margin");
    const lblFatigueLimit = document.querySelector("#lbl-fatigue-limit");
    const predLoader = document.querySelector("#pred-loader");
    const panelVerdict = document.querySelector("#panel-verdict");

    // --- Tab Switching Logic ---
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            // Toggle active classes
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // --- Format Helper ---
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    // --- Initialize Sliders & Values Sync ---
    function initSlidersSync() {
        sliders.forEach(slide => {
            const valSpan = document.getElementById(`val-${slide.id.replace("slide-", "")}`);
            
            slide.addEventListener("input", (e) => {
                let val = parseFloat(e.target.value);
                
                // Extra formatting for specific fields
                if (slide.id === "slide-fatigue") {
                    valSpan.textContent = formatNumber(parseInt(val));
                } else if (slide.id === "slide-p" || slide.id === "slide-s" || slide.id === "slide-nb") {
                    valSpan.textContent = val.toFixed(4);
                } else if (slide.id === "slide-thickness") {
                    valSpan.textContent = val.toFixed(3);
                } else {
                    valSpan.textContent = val.toFixed(1);
                }
                
                // Trigger reactive dynamic predictions with a brief debounce
                debouncePrediction();
            });
        });

        // Trigger predictor on checkbox changes
        chkboxes.forEach(chk => {
            chk.addEventListener("change", debouncePrediction);
        });

        // Trigger predictor on supplier dropdown change
        selSupplier.addEventListener("change", debouncePrediction);
    }

    // --- Debounced API Model Prediction ---
    function debouncePrediction() {
        clearTimeout(predictTimeout);
        predictTimeout = setTimeout(runPartPrediction, 120);
    }

    // --- Fetch Suppliers Scorecard ---
    async function loadSupplierData() {
        try {
            const resp = await fetch("/api/suppliers");
            if (!resp.ok) throw new Error("Failed to load supplier API.");
            
            supplierData = await resp.json();
            
            // 1. Populate Table
            tblSuppliersBody.innerHTML = "";
            supplierData.forEach(sup => {
                let badgeClass = "pass";
                let statusText = "Approved";
                
                if (sup.Supplier_Score < 60) {
                    badgeClass = "fail";
                    statusText = "Critical Audit";
                } else if (sup.Supplier_Score < 80) {
                    badgeClass = "warning";
                    statusText = "Observation";
                }
                
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td><strong>#${sup.Rank}</strong></td>
                    <td>${sup.Supplier_Name}</td>
                    <td><code>${sup.Supplier_ID}</code></td>
                    <td>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <strong>${sup.Supplier_Score.toFixed(1)}</strong>
                            <div style="background: rgba(255,255,255,0.05); width: 60px; height: 6px; border-radius: 99px; overflow:hidden;">
                                <div style="width: ${sup.Supplier_Score}%; height: 100%; background: ${sup.Supplier_Score >= 80 ? 'var(--green)' : (sup.Supplier_Score >= 60 ? 'var(--gold)' : 'var(--red)')};"></div>
                            </div>
                        </div>
                    </td>
                    <td><span class="text-red">${sup.Fail_Rate.toFixed(2)}%</span></td>
                    <td>${sup.Part_Count} batches</td>
                    <td><span class="badge ${badgeClass}">${statusText}</span></td>
                `;
                tblSuppliersBody.appendChild(row);
            });

            // 2. Populate Dropdown Selector in Simulator
            selSupplier.innerHTML = "";
            supplierData.forEach(sup => {
                const opt = document.createElement("option");
                opt.value = sup.Supplier_ID;
                opt.textContent = `${sup.Supplier_Name} (Rank #${sup.Rank})`;
                selSupplier.appendChild(opt);
            });

            // 3. Render Supplier Score Charts
            renderSupplierCharts();
            
        } catch (err) {
            console.error(err);
            tblSuppliersBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-red">
                        <i class="fa-solid fa-triangle-exclamation"></i> API Error loading supplier metrics.
                    </td>
                </tr>
            `;
        }
    }

    // --- Render Scorecard Charts ---
    function renderSupplierCharts() {
        const names = supplierData.map(s => s.Supplier_Name.split(" ")[0]);
        const scores = supplierData.map(s => s.Supplier_Score);
        const rejects = supplierData.map(s => s.Rejected_Count);
        
        // Destroy existing if any
        if (scoresChart) scoresChart.destroy();
        if (rejectionsChart) rejectionsChart.destroy();

        const colors = scores.map(s => s >= 80 ? 'hsl(142, 72%, 50%)' : (s >= 60 ? 'hsl(45, 93%, 47%)' : 'hsl(350, 89%, 60%)'));

        // Bar Chart
        const ctxBar = document.getElementById("chart-scores-bar").getContext("2d");
        scoresChart = new Chart(ctxBar, {
            type: "bar",
            data: {
                labels: names,
                datasets: [{
                    label: "Supplier Quality Rating",
                    data: scores,
                    backgroundColor: colors,
                    borderWidth: 0,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#0d1321",
                        borderColor: "rgba(255,255,255,0.08)",
                        borderWidth: 1,
                        titleColor: "#00f2fe"
                    }
                },
                scales: {
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" },
                        min: 0,
                        max: 100
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8" }
                    }
                }
            }
        });

        // Pie Chart
        const ctxPie = document.getElementById("chart-rejections-pie").getContext("2d");
        rejectionsChart = new Chart(ctxPie, {
            type: "doughnut",
            data: {
                labels: names,
                datasets: [{
                    data: rejects,
                    backgroundColor: [
                        "#00E5FF", "#3b82f6", "#6366f1", "#a855f7", "#ec4899", 
                        "#f43f5e", "#f97316", "#eab308", "#10b981", "#14b8a6"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { color: "#94a3b8", font: { size: 11 } }
                    },
                    tooltip: {
                        backgroundColor: "#0d1321",
                        borderColor: "rgba(255,255,255,0.08)",
                        borderWidth: 1
                    }
                },
                cutout: "60%"
            }
        });
    }

    // --- Fetch Model Metadata & Importance Weights ---
    async function loadMetadata() {
        try {
            const resp = await fetch("/api/metadata");
            if (!resp.ok) throw new Error("Metadata API error");
            
            metaData = await resp.json();
            
            // Set header performance metrics
            document.getElementById("lbl-best-model").textContent = metaData.best_model;
            document.getElementById("lbl-cv-auc").textContent = metaData.cv_roc_auc.toFixed(4);

            // Populate importance charts
            renderImportanceCharts();
            
        } catch (err) {
            console.error("Error loading model metadata:", err);
        }
    }

    // --- Render Feature Coefficients Horizontal Bar Chart ---
    function renderImportanceCharts() {
        const impData = metaData.feature_importances;
        if (!impData || Object.keys(impData).length === 0) return;

        // Sort features by magnitude of coefficient
        const sortedFeatures = Object.keys(impData)
            .map(key => ({ name: key, val: impData[key] }))
            .sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

        const names = sortedFeatures.map(f => f.name);
        const weights = sortedFeatures.map(f => f.val);
        const barColors = weights.map(w => w > 0 ? "rgba(239, 68, 68, 0.75)" : "rgba(16, 185, 129, 0.75)");

        if (featuresChart) featuresChart.destroy();

        const ctxFeatures = document.getElementById("chart-features-importance").getContext("2d");
        featuresChart = new Chart(ctxFeatures, {
            type: "bar",
            data: {
                labels: names,
                datasets: [{
                    label: "Logistic Regression Weight (Influence)",
                    data: weights,
                    backgroundColor: barColors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#0d1321",
                        borderColor: "rgba(255,255,255,0.08)",
                        borderWidth: 1,
                        callbacks: {
                            label: function(ctx) {
                                const w = ctx.raw;
                                return w > 0 
                                    ? `Increases Failure Risk by +${w.toFixed(2)}` 
                                    : `Decreases Failure Risk by ${w.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8", font: { size: 10 } }
                    }
                }
            }
        });
    }

    // --- Run Live Part Rejection Failure Prediction ---
    async function runPartPrediction() {
        predLoader.style.display = "block";
        
        // Assemble payload matching model expected columns
        const payload = {
            "Supplier_ID": selSupplier.value,
            "UTS_MPa": parseFloat(document.getElementById("slide-uts").value),
            "YS_MPa": parseFloat(document.getElementById("slide-ys").value),
            "Elongation_%": parseFloat(document.getElementById("slide-elongation").value),
            "n_value": parseFloat(document.getElementById("slide-n").value),
            "r_value": parseFloat(document.getElementById("slide-r").value),
            "Hardness_HV": parseFloat(document.getElementById("slide-hardness-hv").value),
            "Hardness_HRB": parseFloat(document.getElementById("slide-hardness-hrb").value),
            "Thickness_mm": parseFloat(document.getElementById("slide-thickness").value),
            "Surface_Roughness_Ra_um": parseFloat(document.getElementById("slide-roughness").value),
            "Zinc_Coating_gsm": parseFloat(document.getElementById("slide-coating").value),
            "Salt_Spray_hrs": parseFloat(document.getElementById("slide-salt").value),
            "C_wt%": parseFloat(document.getElementById("slide-c").value),
            "Mn_wt%": parseFloat(document.getElementById("slide-mn").value),
            "Si_wt%": parseFloat(document.getElementById("slide-si").value),
            "P_wt%": parseFloat(document.getElementById("slide-p").value),
            "S_wt%": parseFloat(document.getElementById("slide-s").value),
            "Cr_wt%": parseFloat(document.getElementById("slide-cr").value),
            "Nb_wt%": parseFloat(document.getElementById("slide-nb").value),
            "Bend_Ratio_d/t": 1.0,  // fallback constant
            "Weld_Nugget_Dia_mm": parseFloat(document.getElementById("slide-weld").value),
            "Fatigue_Cycles": parseFloat(document.getElementById("slide-fatigue").value),
            "Charpy_Impact_J": parseFloat(document.getElementById("slide-charpy").value),
            
            // Checkboxes
            "Thickness_Tol_OK_bin": document.getElementById("chk-thick-ok").checked ? 1.0 : 0.0,
            "Coating_OK_bin": document.getElementById("chk-coating-ok").checked ? 1.0 : 0.0,
            "Salt_Spray_OK_bin": document.getElementById("chk-salt-ok").checked ? 1.0 : 0.0,
            "Composition_OK_bin": document.getElementById("chk-comp-ok").checked ? 1.0 : 0.0,
            "Bend_Test_OK_bin": document.getElementById("chk-bend-ok").checked ? 1.0 : 0.0,
            "Weld_OK_bin": document.getElementById("chk-weld-ok").checked ? 1.0 : 0.0,
            "Dimensional_OK_bin": document.getElementById("chk-dim-ok").checked ? 1.0 : 0.0,
            "Charpy_OK_bin": document.getElementById("chk-charpy-ok").checked ? 1.0 : 0.0
        };

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error("Predict request failed");
            const result = await resp.json();

            // Update UI components with result
            const prob = result.fail_probability;
            const verdict = result.verdict;
            
            // 1. Percentage Display
            lblProbability.textContent = `${(prob * 100).toFixed(2)}%`;
            lblMargin.textContent = `${((1 - prob) * 100).toFixed(2)}%`;
            
            // 2. Adjust Gauge Ring
            // 126 is the stroke-dasharray (semi-circle). Max out of 126 dash offset.
            const offset = 126 - (prob * 126);
            gaugeFill.style.strokeDashoffset = offset;
            
            // Set Gauge color dynamically
            if (prob < 0.05) {
                gaugeFill.style.stroke = "var(--green)";
                badgeVerdict.className = "verdict-badge status-pass";
                badgeVerdict.textContent = "PASS";
                lblMargin.className = "text-green";
                panelVerdict.style.borderColor = "var(--border-glass)";
            } else if (prob < 0.25) {
                gaugeFill.style.stroke = "var(--gold)";
                badgeVerdict.className = "verdict-badge status-pass"; // Still pass technically, but caution
                badgeVerdict.style.color = "var(--gold)";
                badgeVerdict.style.textShadow = "0 0 15px rgba(217, 119, 6, 0.35)";
                badgeVerdict.textContent = "CAUTION";
                lblMargin.className = "text-gold";
                panelVerdict.style.borderColor = "var(--gold)";
            } else {
                gaugeFill.style.stroke = "var(--red)";
                badgeVerdict.className = "verdict-badge status-fail";
                badgeVerdict.textContent = "FAIL";
                lblMargin.className = "text-red";
                panelVerdict.style.borderColor = "var(--red)";
            }

            // 3. Informative Text Feedback
            if (verdict === "PASS") {
                if (prob < 0.05) {
                    lblExplanation.textContent = "Excellent rating! All tensile, chemical weights, and fatigue limits are fully optimal. Part presents negligible structural risk.";
                } else {
                    lblExplanation.textContent = "Part meets minimum safety tolerances, but elevated roughness or sub-optimal hardness requires caution.";
                }
            } else {
                lblExplanation.textContent = "Part rejected. Excessive deviation in tensile properties or chemical composition (elevated Sulfur/Phosphorus or low Carbon) fails standard B-pillar safety audits.";
            }

            // 4. Update fatigue limit helper
            lblFatigueLimit.textContent = payload.Fatigue_Cycles >= 1000000 ? "OPTIMAL" : "SUB-OPTIMAL";
            lblFatigueLimit.className = payload.Fatigue_Cycles >= 1000000 ? "text-green" : "text-gold";

        } catch (err) {
            console.error("Prediction error:", err);
        } finally {
            predLoader.style.display = "none";
        }
    }

    // --- Init Pipeline ---
    async function init() {
        initSlidersSync();
        
        // Core loading calls
        await loadSupplierData();
        await loadMetadata();
        
        // Execute initial prediction on default values
        runPartPrediction();
    }

    init();
});
