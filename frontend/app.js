const API_BASE_URL = window.API_BASE_URL || "http://localhost:8000";
const form = document.getElementById("churn-form");
const resultValue = document.querySelector(".result__value");
const resultDetail = document.querySelector(".result__detail");

const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultValue.textContent = "Running prediction...";
  resultDetail.textContent = "Contacting the API.";

  const formData = new FormData(form);
  const payload = {
    gender: formData.get("gender"),
    senior_citizen: Number(formData.get("seniorCitizen")),
    partner: formData.get("partner"),
    dependents: formData.get("dependents"),
    tenure: Number(formData.get("tenure")),
    phone_service: formData.get("phoneService"),
    multiple_lines: formData.get("multipleLines"),
    internet_service: formData.get("internetService"),
    online_security: formData.get("onlineSecurity"),
    online_backup: formData.get("onlineBackup"),
    device_protection: formData.get("deviceProtection"),
    tech_support: formData.get("techSupport"),
    streaming_tv: formData.get("streamingTv"),
    streaming_movies: formData.get("streamingMovies"),
    contract: formData.get("contract"),
    paperless_billing: formData.get("paperlessBilling"),
    payment_method: formData.get("paymentMethod"),
    monthly_charges: Number(formData.get("monthlyCharges")),
    total_charges: Number(formData.get("totalCharges")),
  };

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Prediction failed.");
    }

    const data = await response.json();
    const label = data.churn_label === 1 ? "Churn Risk" : "Healthy";
    resultValue.textContent = label;

    if (data.churn_probability !== null && data.churn_probability !== undefined) {
      resultDetail.textContent = `Churn probability: ${formatPercent(
        data.churn_probability
      )}`;
    } else {
      resultDetail.textContent = "Probability not available.";
    }
  } catch (error) {
    resultValue.textContent = "Prediction error";
    resultDetail.textContent = error.message;
  }
});
