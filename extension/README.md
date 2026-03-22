# MindGuard Chrome Extension 🧩

This Chrome Extension allows you to directly analyze text from any webpage (like social media, forums, or emails) using your local MindGuard Community Analyzer.

## How to Install (Developer Mode)

1. Open Google Chrome.
2. In the URL bar, type: `chrome://extensions/` and hit Enter.
3. In the top right corner, toggle **Developer mode** to ON.
4. Click the **Load unpacked** button in the top left.
5. In the file dialog, navigate to your project folder and select the `extension` folder (`d:\mental crisis detenection\extension`).
6. The MindGuard Analyzer icon will appear in your extensions toolbar!

## How to Use

1. Ensure the MindGuard API is running in the background. If you haven't started it, run the dashboard and click the "Start Local API Server" button in the "System API" tab.
    ```bash
    # Ensure this is running in your terminal:
    python app/community_dashboard.py
    ```
2. Highlight any text on a webpage (e.g., a Reddit post or a forum comment).
3. Click the MindGuard extension icon in the top right of your browser.
4. Click **Grab Selected Text** to instantly pull in what you highlighted.
5. Click **Analyze** to get the 8-dimensional emotional profile and crisis severity.

**Privacy Note:** The extension sends data locally to your own machine at `http://127.0.0.1:8000`. No data leaves your computer.
