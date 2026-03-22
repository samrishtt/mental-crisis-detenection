document.addEventListener('DOMContentLoaded', () => {
  const inputText = document.getElementById('inputText');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const grabBtn = document.getElementById('grabBtn');
  
  const loading = document.getElementById('loading');
  const errorMsg = document.getElementById('errorMsg');
  const results = document.getElementById('results');
  
  const coreResult = document.getElementById('coreResult');
  const flagsContainer = document.getElementById('flagsContainer');
  const emotionsList = document.getElementById('emotionsList');

  // Grab selected text from the active tab
  grabBtn.addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection().toString()
      });
      
      if (results && results[0] && results[0].result) {
        inputText.value = results[0].result;
      } else {
        inputText.value = "No text selected on page.";
      }
    } catch (e) {
      console.error(e);
      inputText.value = "Cannot read text from this page. Note: Chrome extensions block script execution on chrome:// URLs.";
    }
  });

  // Send request to local API
  analyzeBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (text.length < 10) {
      showError("Please enter at least 10 characters to analyze.");
      return;
    }

    loading.classList.remove('hidden');
    errorMsg.classList.add('hidden');
    results.classList.add('hidden');
    
    // Reset contents
    coreResult.innerHTML = '';
    flagsContainer.innerHTML = '';
    emotionsList.innerHTML = '';

    try {
      // Connect to the local FastAPI server
      const response = await fetch('http://127.0.0.1:8000/api/v1/analyze/single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'demo-organization-key' // Replace if required by config
        },
        body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }

      const data = await response.json();
      renderResults(data.analysis);
      
    } catch (e) {
      console.error(e);
      showError("Could not connect to local MindGuard API. Please ensure your Python backend is running (python app/community_dashboard.py).");
    } finally {
      loading.classList.add('hidden');
    }
  });

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
  }

  function renderResults(analysis) {
    results.classList.remove('hidden');

    const core = analysis.core_severity;
    
    // Core severity styling
    let coreColor = "#4CAF50";
    if (core.level === "Severe Crisis") coreColor = "#D32F2F";
    else if (core.level === "Moderate") coreColor = "#FF9800";
    else if (core.level === "Mild") coreColor = "#FFC107";

    coreResult.style.borderLeftColor = coreColor;
    coreResult.innerHTML = `
      <div class="result-level" style="color: ${coreColor}">${core.level}</div>
      <div class="result-conf">Confidence: ${(core.confidence * 100).toFixed(0)}%</div>
    `;

    // Flags
    if (analysis.flags && analysis.flags.length > 0) {
      analysis.flags.forEach(f => {
        const div = document.createElement('div');
        div.className = `flag ${f.type}`;
        div.textContent = `${f.type.toUpperCase()}: ${f.message}`;
        flagsContainer.appendChild(div);
      });
    }

    // Emotions
    const dims = analysis.dimensions;
    for (const key in dims) {
      const d = dims[key];
      const div = document.createElement('div');
      div.className = 'dim';
      
      div.innerHTML = `
        <div class="dim-name">
          <span>${d.icon}</span> <span>${d.label}</span>
        </div>
        <div class="dim-bar-wrap">
          <div class="dim-bar" style="width: ${d.score}%; background: ${d.color}; opacity: 0.8;"></div>
        </div>
        <div class="dim-val">${d.score}</div>
      `;
      emotionsList.appendChild(div);
    }
  }
});
