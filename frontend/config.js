// frontend/config.js
// The ONE place that decides which API the pages talk to, plus the small
// helpers every page shares (missing-input check, friendly error messages).

// Public API on Render (use this for the Render demo):
const API_ORIGIN = "https://smm-202604-c4-nyaga.onrender.com";

// Other targets -- comment the line above and uncomment ONE of these:
// const API_ORIGIN = "http://127.0.0.1:5000";   // python api.py (Flask dev server)
// const API_ORIGIN = "https://localhost";        // Docker + Nginx (self-signed certificate)


// Returns the labels of required fields that are still empty.
function missingFields(ids) {
  return ids
    .filter(id => document.getElementById(id).value.trim() === "")
    .map(id => {
      const label = document.querySelector(`label[for="${id}"]`);
      return label ? label.childNodes[0].textContent.trim() : id;
    });
}

// Number from an input, or null when the box is empty.
function numberOrNull(id) {
  const raw = document.getElementById(id).value.trim();
  return raw === "" ? null : Number(raw);
}

// POSTs JSON to the API and returns the parsed answer. Throws an Error whose
// message is safe to show to the user (API validation message or a clear
// "could not reach the API" text).
async function postJson(path, payload) {
  let response;
  try {
    response = await fetch(API_ORIGIN + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (networkError) {
    throw new Error(
      `Could not reach the API at ${API_ORIGIN}. Check that it is running ` +
      `(Render's free tier can take up to a minute to wake up).`
    );
  }

  let data = null;
  try { data = await response.json(); } catch (ignored) { /* not JSON */ }

  if (!response.ok) {
    const message = data && data.error ? data.error : `Server responded ${response.status}`;
    const details = data && data.details ? ` (${data.details})` : "";
    throw new Error(message + details);
  }
  return data;
}

function showError(resultBox, message) {
  resultBox.className = "show err";
  resultBox.textContent = message;
}
