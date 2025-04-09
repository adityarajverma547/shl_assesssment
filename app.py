import gradio as gr
import requests

API_URL = "https://apiend-08av.onrender.com/recommend"

def fetch_recommendations(query, top_k):
    try:
        payload = {"query": query}
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        data = response.json()

        # New response format key
        recommendations = data.get("recommended_assessments", [])
        if not isinstance(recommendations, list):
            return "❌ Unexpected response format from API."

        formatted_results = ""
        for i, rec in enumerate(recommendations[:int(top_k)], start=1):
            title = rec.get("description", "N/A")
            link = rec.get("url", "#")
            adaptive = rec.get("adaptive_support", "N/A")
            remote = rec.get("remote_support", "N/A")
            test_types = rec.get("test_type", [])
            duration = rec.get("duration", "N/A")

            test_type_list = ", ".join(test_types) if isinstance(test_types, list) else test_types

            formatted_results += (
                f"### {i}. [{title}]({link})\n"
                f"- **Remote Testing**: {remote}\n"
                f"- **Adaptive Support**: {adaptive}\n"
                f"- **Test Types**: {test_type_list}\n"
                f"- **Duration**: {duration} minutes\n\n"
            )

        return formatted_results.strip()

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
gr.Interface(
    fn=fetch_recommendations,
    inputs=[
        gr.Textbox(label="Enter your query"),
        gr.Number(label="How many recommendations?", value=5, precision=0),
    ],
    outputs=gr.Markdown(label="Top SHL Assessment Recommendations"),
    title="SHL Assessment Recommender",
    description="Get recommended SHL assessments based on your job role or skills.",
).launch(share=True)
