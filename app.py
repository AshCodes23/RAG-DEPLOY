from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

load_dotenv()
from src.rag_core import RAGSystem

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/tmp/docs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

rag = RAGSystem()  # models load ONCE

@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":

        # Upload document
        if "document" in request.files:
            file = request.files["document"]
            if file and file.filename.endswith(".pdf"):
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(save_path)
                rag.rebuild_index()
                message = "Document uploaded and indexed."

        # Ask question
        elif "question" in request.form:
            question = request.form["question"].strip()
            if question:
                # VERY IMPORTANT: blocking call
                rag.ask(question)

    return render_template(
        "index.html",
        history=rag.conversation_memory,
        message=message
    )



if __name__ == "__main__":
    app.run(debug=True, port=5001)
