const API_URL =
  window.DEVDOCS_API_URL ||
  `http://${window.location.hostname}:8000`;

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const chatHistory = document.querySelector("#chat-history");
const newChatButton = document.querySelector("#new-chat");

let lastQuestion = "";
let isLoading = false;


/* =========================================================
   SUBMIT
   ========================================================= */

form.addEventListener("submit", (event) => {
  event.preventDefault();

  submitQuestion(questionInput.value);
});


/* Enter = send
   Shift + Enter = new line
*/

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();

    form.requestSubmit();
  }
});


/* Automatically grow textarea */

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";

  questionInput.style.height =
    Math.min(questionInput.scrollHeight, 180) + "px";
});


/* =========================================================
   NEW CHAT
   ========================================================= */

newChatButton.addEventListener("click", () => {
  chatHistory.innerHTML = `
    <section id="welcome" class="welcome">
      <div class="welcome-icon">D</div>

      <h1>How can I help?</h1>

      <p>
        Ask a question about the indexed documentation.
      </p>
    </section>
  `;

  questionInput.value = "";

  resetTextarea();

  lastQuestion = "";

  questionInput.focus();
});


/* =========================================================
   ASK QUESTION
   ========================================================= */

async function submitQuestion(value) {
  const query = value.trim();

  if (!query || isLoading) {
    return;
  }

  lastQuestion = query;

  removeWelcome();

  addUserMessage(query);

  questionInput.value = "";

  resetTextarea();

  setLoading(true);

  try {
    const response = await fetch(
      `${API_URL}/api/v1/rag/query`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          query,
        }),
      }
    );

    const data = await response
      .json()
      .catch(() => null);

    if (!response.ok) {
      throw new Error(
        `The API returned HTTP ${response.status}.`
      );
    }

    if (!data || data.ok === false) {
      throw new Error(
        data?.error ||
        "The documentation assistant could not answer this question."
      );
    }

    if (!data.answer || !data.answer.trim()) {
      throw new Error(
        "The assistant returned an empty answer."
      );
    }

    removeLoading();

    addAssistantMessage(data);

  } catch (error) {

    removeLoading();

    addErrorMessage(
      error instanceof TypeError
        ? "The backend could not be reached. Make sure the API is running."
        : error.message
    );

  } finally {

    setLoading(false);

    questionInput.focus();
  }
}


/* =========================================================
   USER MESSAGE
   ========================================================= */

function addUserMessage(text) {

  const message = document.createElement("div");

  message.className = "message user";

  message.innerHTML = `
    <div class="user-message">
      ${escapeHtml(text)}
    </div>
  `;

  chatHistory.appendChild(message);

  scrollToBottom();
}


/* =========================================================
   ASSISTANT MESSAGE
   ========================================================= */

function addAssistantMessage(data) {

  const message = document.createElement("div");

  message.className = "message assistant";

  const sources = Array.isArray(data.citations)
    ? data.citations
    : [];

  message.innerHTML = `
    <div class="assistant-icon">D</div>

    <div class="assistant-content">

      <div class="answer">
        ${renderMarkdown(data.answer)}
      </div>

      ${
        sources.length
          ? renderSources(sources)
          : ""
      }

      <div class="message-meta">

        <span>
          ${Number(data.chunks_in_context || 0)} sources
        </span>

        <button
          type="button"
          class="copy-answer"
          aria-label="Copy answer"
        >
          Copy
        </button>

      </div>

    </div>
  `;

  chatHistory.appendChild(message);

  const copyButton =
    message.querySelector(".copy-answer");

  copyButton.addEventListener("click", async () => {

    try {

      await navigator.clipboard.writeText(
        data.answer
      );

      copyButton.textContent = "Copied";

      setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1500);

    } catch {
      copyButton.textContent = "Copy failed";
    }

  });

  scrollToBottom();
}


/* =========================================================
   SOURCES
   ========================================================= */

function renderSources(sources) {

  const links = sources
    .map((citation) => {

      const url = safeUrl(citation.url);

      if (!url) {
        return "";
      }

      const title = escapeHtml(
        citation.title ||
        citation.source_id ||
        "Documentation"
      );

      const sourceId = escapeHtml(
        citation.source_id || ""
      );

      return `
        <a
          class="source-link"
          href="${escapeHtml(url)}"
          target="_blank"
          rel="noopener noreferrer"
        >
          ${title}
          ${sourceId ? ` · ${sourceId}` : ""}
        </a>
      `;
    })
    .join("");

  if (!links) {
    return "";
  }

  return `
    <div class="sources">

      <div class="sources-title">
        Sources
      </div>

      ${links}

    </div>
  `;
}


/* =========================================================
   LOADING
   ========================================================= */

function setLoading(value) {

  isLoading = value;

  sendButton.disabled = value;

  if (value) {

    const loading = document.createElement("div");

    loading.id = "loading-message";

    loading.className = "loading-message";

    loading.innerHTML = `
      <div class="assistant-icon">D</div>

      <div class="loading-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;

    chatHistory.appendChild(loading);

    scrollToBottom();
  }
}


function removeLoading() {

  const loading =
    document.querySelector("#loading-message");

  if (loading) {
    loading.remove();
  }
}


/* =========================================================
   ERROR
   ========================================================= */

function addErrorMessage(message) {

  const wrapper =
    document.createElement("div");

  wrapper.className =
    "message assistant";

  wrapper.innerHTML = `
    <div class="assistant-icon">D</div>

    <div class="assistant-content">

      <div class="error-message">
        ${escapeHtml(message)}
      </div>

      <div class="message-meta">

        <button
          type="button"
          id="retry-button"
        >
          Try again
        </button>

      </div>

    </div>
  `;

  chatHistory.appendChild(wrapper);

  wrapper
    .querySelector("#retry-button")
    .addEventListener("click", () => {

      submitQuestion(lastQuestion);

    });

  scrollToBottom();
}


/* =========================================================
   MARKDOWN
   ========================================================= */

function renderMarkdown(markdown) {

  let html = escapeHtml(markdown);


  /* Code blocks */

  html = html.replace(
    /```([\w+-]*)\n?([\s\S]*?)```/g,
    (_, language, code) => {

      const cleanCode =
        code.trim();

      return `
        <div class="code-block">

          <div class="code-header">

            <span>
              ${escapeHtml(language || "code")}
            </span>

            <button
              type="button"
              class="copy-code"
              onclick="copyCode(this)"
            >
              Copy code
            </button>

          </div>

          <pre>${cleanCode}</pre>

        </div>
      `;
    }
  );


  /* Headings */

  html = html.replace(
    /^### (.+)$/gm,
    "<p><strong>$1</strong></p>"
  );

  html = html.replace(
    /^## (.+)$/gm,
    "<p><strong>$1</strong></p>"
  );

  html = html.replace(
    /^# (.+)$/gm,
    "<p><strong>$1</strong></p>"
  );


  /* Bold */

  html = html.replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>"
  );


  /* Inline code */

  html = html.replace(
    /`([^`]+)`/g,
    "<code>$1</code>"
  );


  /* Markdown links */

  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    (_, label, url) => {

      const safe = safeUrl(url);

      if (!safe) {
        return escapeHtml(label);
      }

      return `
        <a
          href="${escapeHtml(safe)}"
          target="_blank"
          rel="noopener noreferrer"
        >
          ${label}
        </a>
      `;
    }
  );


  /* Unordered lists */

  html = html.replace(
    /(?:^|\n)- (.+)/g,
    "<li>$1</li>"
  );

  html = html.replace(
    /(<li>.*<\/li>)/gs,
    "<ul>$1</ul>"
  );


  /* Paragraphs */

  const blocks =
    html.split(/\n\s*\n/);

  return blocks
    .map((block) => {

      block = block.trim();

      if (!block) {
        return "";
      }

      if (
        block.startsWith("<div class=\"code-block\">") ||
        block.startsWith("<ul>") ||
        block.startsWith("<p>")
      ) {
        return block;
      }

      return `
        <p>
          ${block.replace(/\n/g, "<br>")}
        </p>
      `;

    })
    .join("");
}


/* =========================================================
   COPY CODE
   ========================================================= */

async function copyCode(button) {

  const code =
    button
      .closest(".code-block")
      .querySelector("pre")
      .innerText;

  try {

    await navigator.clipboard.writeText(code);

    button.textContent = "Copied";

    setTimeout(() => {
      button.textContent = "Copy code";
    }, 1500);

  } catch {

    button.textContent = "Copy failed";

  }
}


/* =========================================================
   HELPERS
   ========================================================= */

function removeWelcome() {

  const welcome =
    document.querySelector("#welcome");

  if (welcome) {
    welcome.remove();
  }
}


function resetTextarea() {

  questionInput.style.height = "auto";
}


function scrollToBottom() {

  requestAnimationFrame(() => {

    const chatPage =
      document.querySelector(".chat-page");

    chatPage.scrollTop =
      chatPage.scrollHeight;

  });
}


function safeUrl(value) {

  try {

    const url = new URL(value);

    if (
      url.protocol === "https:" ||
      url.protocol === "http:"
    ) {
      return url.href;
    }

    return "";

  } catch {

    return "";
  }
}


function escapeHtml(value) {

  return String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[character]
  );
}