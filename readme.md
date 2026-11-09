# Anlytx AI: A Quick First Look at Your Data

Upload a spreadsheet, pick what you want to predict, and within a couple of minutes
get a **preliminary quick report**: what was cleaned, which models worked best, and
how accurate they are. Then **chat with the results** in plain English. No coding needed.

> This is a **first-look baseline**, not a final answer: default model settings, no tuning.
> It helps you decide quickly whether your data is worth a deeper analysis.

## How to use it

1. Open the app link: **[YOUR N8N FORM LINK]**
2. Upload a CSV or Excel file and choose the task:
   - **Regression**: predict a number (price, sales, age...)
   - **Classification**: predict a category (yes/no, positive/negative...)
3. Pick the **target column** (the one you want to predict). If your data has a
   **date/time column**, pick it too (it will be removed); otherwise leave it as **None**.
   Then click **Submit & Run Pipeline**.
4. Read the report, then choose what to do next:
   - **Ask a question**, for example:
     - "How many missing values were there?"
     - "Which model was best and why?"
     - "Which columns were removed or encoded?"
     - "What are the limitations of these results?"
   - **End chat** if you only wanted the report.
5. After each answer, choose **Ask a question** again or **End chat** when you're done.

## What it does automatically

- Converts number-like text (`4,25,000`, `45,000 kms`) into real numbers
- Removes rows with an empty target and fills other missing values
- Drops the date column you picked, and also catches leftover date/time columns by itself
- Drops ID-like columns (row numbers, customer or account codes) by looking at the sample values
- Drops useless text columns (names, links, hashes) and turns review text into a sentiment score (English only)
- Encodes text categories, caps extreme values, scales features, removes duplicated information
- Trains and compares 3 models (RandomForest, GradientBoosting, and Ridge or LogisticRegression)
- Handles imbalanced classes with class weighting and shows per-class results
- If the target is very skewed (like prices), trains on its log and converts results back
- Stops with a clear error if a classification target has too many unique values (more than 20),
  because it looks like a regression target

## How it works

```
Browser -> n8n forms -> Python backend (FastMCP on Horizon)
                          |- Code inspects the data (columns, sample rows, unique ratios)
                          |- AI agent picks only the optional cleaning tools
                          |    (ID columns, leftover date columns, text/sentiment, irrelevant columns)
                          |- Code always runs the compulsory steps (cleanup, encoding, scaling, training)
                          |- Each tool records what it did in report.json
                          |- 3 models are trained, best chosen by cross-validation
                          v
          Summary built from report.json  ->  Chat assistant (Groq) answers questions
```

Every step writes its facts into `report.json`. The summary is built from it directly
(no AI guessing), and the chat assistant reads it: facts about your run come from the
report, and it may add general explanations that are marked as general knowledge.

## Setup (project owner)

**Needs:** a Horizon account, an n8n instance, and a free [Groq](https://console.groq.com) API key.

```bash
git clone [YOUR REPO LINK]
cd Anlytx_AI
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Create a `.env` file (never commit it):
```
GROQ_API_KEY=your_key_here
```

Run locally with `python main.py`, or deploy by pushing to Horizon.
Your MCP server link: **[YOUR HORIZON LINK]**

**n8n:** open the workflow **[YOUR WORKFLOW LINK]**, add your Groq key to the chat model nodes,
point both HTTP Request nodes to your Horizon link, set the pipeline node timeout to
`600000` ms, then activate the workflow and share the production form link.

## Known limitations

- **Preliminary only:** default settings, no hyperparameter tuning.
- **Slight data leakage:** scaling, outlier limits and feature selection are fitted on the
  full dataset before the train/test split, so test scores may be slightly optimistic.
  The fix is a scikit-learn Pipeline fitted on training rows only.
- **AI-based column detection:** ID and leftover date columns are judged by an AI model
  from sample values. It keeps a column when unsure, so it may miss one, but it should not
  drop useful columns. Picking the date column yourself is the most reliable option.
- **Free-tier AI limits:** Groq's free plan has token limits. Retries and fallback models
  are built in, but heavy use can still hit them.
- **Temporary storage:** files live in `/tmp` on the server and can be wiped on restart.
- Works on tabular data only (not images or audio).

## Roadmap

Fit preprocessing on training data only, add tuning and more models, feature importance,
show chat history, persist reports, a custom web UI, and offer the cleaned dataset for download.

**Built with:** Python, FastMCP, LangChain, Groq, scikit-learn, pandas, VADER, n8n, Horizon.