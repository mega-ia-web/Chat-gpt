// Toggle dark mode
document.getElementById("theme-toggle").addEventListener("click", () => {
  document.body.dataset.theme =
    document.body.dataset.theme === "dark" ? "light" : "dark";
});

// Exemple appel API IA
document.getElementById("call-ai").addEventListener("click", async () => {
  alert("Appel API IA simulé 🚀");
  // Ici tu pourras mettre ton fetch() vers OpenRouter ou Gemini
});
