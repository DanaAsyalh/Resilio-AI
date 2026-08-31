#  Resilio AI
**Because infrastructure should heal itself before the pager goes off.**

[![IBM AI Builders](https://img.shields.io/badge/IBM_Global_AI_Builders-Wildcard_Track-0f62fe?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](#)

![Resilio AI Dashboard]
<img width="1515" height="876" alt="image" src="https://github.com/user-attachments/assets/953b2d20-7cd4-4fe2-9131-2c2171544a4a" />


##  The Story
Most AIOps tools wait for a server to crash or a metric to cross a red line before they start screaming. By then, the damage is done, users are impacted, and SRE teams are spending hours digging through logs at 3 AM. 

I wanted to change that. **Resilio AI** treats server telemetry like digital signals. Instead of just reacting to sudden spikes, it actively looks for "regime shifts"—the subtle, underlying anomalies that happen *before* a catastrophic failure (like a volumetric DDoS attack or a silent memory leak).

##  What it does
Resilio AI acts as an autonomous, predictive co-worker for your DevOps team:
* **📡 Signal-Based Detection:** Uses `Isolation Forest` machine learning to analyze rolling windows of time-series telemetry (CPU, Memory, Latency, Error Rates).
* ** Instant Root Cause Analysis (RCA):** Doesn't just say "Something is wrong." It tells you *what* is wrong, with a severity and confidence score.
* ** Autonomous Defense:** Automatically generates and executes Tier-1 remediation commands (e.g., Kubernetes network policy drops, Ingress scaling) to absorb the impact instantly.

##  Built With
* **AI & Data:** `scikit-learn`, `pandas`, `numpy`
* **Frontend:** `streamlit` (Custom dark-mode UI)
* **Simulation:** Synthetic time-series generation mimicking live Kubernetes cluster behavior.

##  IBM Bob Collaboration
Building an enterprise-grade AI system in a few days is no joke. I teamed up with **IBM Bob** to handle the heavy lifting. Bob helped me write the synthetic telemetry generation engine (injecting realistic cyclical loads and regime shifts), refine the machine learning thresholds, and structure the Streamlit UI components. *(Check out the `/docs` folder for our brainstorming and prompt history!)*

##  Quick Start (Run it locally)

Want to see the system predict an outage? Run it yourself:

1. **Clone it:**
   ```bash
   git clone [https://github.com/your-username/resilio-ai.git](https://github.com/your-username/resilio-ai.git)
   cd resilio-ai



1)  Install the dependencies:
streamlit run app.py

3)  Fire it up:
streamlit run app.py

3)  Break it (on purpose): Click the "Simulate System Outage" button in the dashboard and watch the AI detect, diagnose, and prescribe remediation in real-time.

 About the Builder
Dana Tariq Asyalh Telecommunication & Electronics Engineering @Taflia Technical University Vice Chairman, IEEE RAS Student Chapter and a chess player!

Passionate about Cloud Infrastructure, Dev-ops, and building AI that solves real backend problems. Let's connect!

   
