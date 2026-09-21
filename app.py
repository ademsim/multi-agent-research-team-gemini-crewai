import re
import time

import requests
import streamlit as st

st.set_page_config(page_title="AI Research Team")

st.title("AI Research Team (Gemini)")
st.write(
    "Three AI agents work one after another on a topic: a **Researcher** collects facts, a **Writer** writes a short "
    "report from those notes, and an **Editor** checks the report and returns the final version. "
    "You can also compare the team with a single prompt to the same model."
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BUSY_CODES = (429, 500, 502, 503, 504)
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

ROLES = {
    "Researcher": "You are a careful research analyst. You only state facts you are confident about, you mark uncertain "
                  "points with '(uncertain)', and you never invent statistics, dates, or sources.",
    "Writer": "You are a clear technical writer who explains things simply and does not add facts that are not in the notes.",
    "Editor": "You are a strict editor. You remove or soften claims that are not supported, and you keep the report short "
              "and well structured.",
}


class ApiError(Exception):
    pass


def friendly_error(status, text, api_key):
    text = text.replace(api_key, "***")
    hints = {
        400: "The request was rejected. Check that the API key is valid.",
        401: "The API key was rejected. Create a Gemini API key at aistudio.google.com/apikey and paste it again.",
        403: "The API key has no permission for this model or service.",
        404: "This model name was not found for your key. Try another model in the list.",
        429: "The free quota or the per-minute limit is used up. Wait a minute and try again.",
        503: "The model is very busy right now. Try again in a moment or choose another model.",
    }
    return f"Gemini API error {status}. {hints.get(status, '')} ({text[:160]})"


def ask(api_key, models, system, prompt, temperature=0.7):
    """One call to Gemini. Retries when the model is busy, then tries the next model in the list."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    response = None
    for model in models:
        for attempt in range(3):
            try:
                response = requests.post(API_URL.format(model=model), headers=headers, json=body, timeout=120)
            except requests.RequestException:
                raise ApiError("Could not reach the Gemini API. Check your internet connection and try again.")
            if response.status_code == 200:
                try:
                    parts = response.json()["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts).strip(), model
                except (KeyError, IndexError):
                    raise ApiError("The model returned no text (the request may have been blocked by a safety filter).")
            if response.status_code not in BUSY_CODES:
                raise ApiError(friendly_error(response.status_code, response.text, api_key))
            time.sleep(2 ** (attempt + 1))
    raise ApiError(friendly_error(response.status_code, response.text, api_key))


def numbers_in(text):
    return len(re.findall(r"\d+(?:\.\d+)?", text))


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", type="password", help="Used only in this session. It is never stored or shown.")
    st.caption("No key yet? Create a free one at [Google AI Studio](https://aistudio.google.com/apikey).")
    model = st.selectbox("Model (if it is busy, the next ones are tried automatically)", MODELS)
    language = st.selectbox("Report language", ["English", "Turkish"])
    compare = st.checkbox("Also run a single prompt for comparison", value=True)

topic = st.text_input("Topic", "Togg, the Turkish electric car")

if st.button("Run the team", type="primary"):
    if not api_key.strip():
        st.warning("Please paste your Gemini API key in the sidebar first.")
        st.stop()
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    key = api_key.strip()
    models = [model] + [m for m in MODELS if m != model]
    result = {"topic": topic.strip()}
    try:
        with st.status("The team is working...", expanded=True) as status:
            start = time.time()
            st.write("🔎 Researcher is collecting facts...")
            result["notes"], m1 = ask(key, models, ROLES["Researcher"],
                                      f"Collect 6 key facts about: {topic}\nReturn a bullet list of 6 short facts. Mark any "
                                      "uncertain fact with '(uncertain)'. Do not invent numbers or sources.")
            st.write("✍️ Writer is writing the report...")
            result["draft"], m2 = ask(key, models, ROLES["Writer"],
                                      f"Research notes:\n{result['notes']}\n\nWrite a report of about 150 words about: {topic}, "
                                      f"using only the facts in the research notes. Write in {language}.")
            st.write("🧐 Editor is checking the report...")
            result["final"], m3 = ask(key, models, ROLES["Editor"],
                                      f"Research notes:\n{result['notes']}\n\nDraft report:\n{result['draft']}\n\n"
                                      "Review the draft against the notes. Remove or soften claims that are not supported by the "
                                      "notes, fix unclear sentences, and return the final report in markdown with a title and a "
                                      f"short 'Limitations' line. Write in {language}. Return only the final report.")
            result["team_seconds"] = time.time() - start
            result["models"] = sorted({m1, m2, m3})

            if compare:
                st.write("⚖️ Running a single prompt for comparison...")
                start = time.time()
                result["single"], _ = ask(key, models, "You are a helpful writer.",
                                          f"Write a report of about 150 words about: {topic}, in {language}, with a title and a "
                                          "short 'Limitations' line. Do not invent statistics or sources.")
                result["single_seconds"] = time.time() - start
            status.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = result
    except ApiError as e:
        st.error(str(e))

result = st.session_state.get("result")
if result:
    st.subheader(f"Final report: {result['topic']}")
    st.markdown(result["final"])
    st.caption("Model used: " + ", ".join(result["models"]))

    with st.expander("What each agent did"):
        st.markdown("**1. Researcher's notes**")
        st.markdown(result["notes"])
        st.markdown("**2. Writer's draft**")
        st.markdown(result["draft"])

    if "single" in result:
        st.subheader("Team vs single prompt")
        st.dataframe(
            {
                "": ["LLM calls", "seconds", "words", "numbers mentioned"],
                "Single prompt": [1, round(result["single_seconds"]), len(result["single"].split()), numbers_in(result["single"])],
                "Team (3 agents)": [3, round(result["team_seconds"]), len(result["final"].split()), numbers_in(result["final"])],
            },
            hide_index=True,
        )
        with st.expander("Single prompt result"):
            st.markdown(result["single"])
        st.caption(
            "More numbers means more detail, but also more chances for a mistake. Check every date and number in a reliable source."
        )

st.info(
    "The agents do not search the web: they only use what the model already knows, which can be outdated or wrong. "
    "An editor agent cannot fix this, because it uses the same model knowledge. Always verify important facts."
)
