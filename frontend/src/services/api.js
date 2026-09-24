/**
 * Centralized API Client for Brain MRI Classification Backend.
 * Communicates with FastAPI endpoints (/health, /model-info, /predict, /explain).
 * Handles network failures, abort timeouts, and structured error responses.
 */

import { API_BASE_URL } from "../utils/constants";

const DEFAULT_TIMEOUT_MS = 60000; // 60s timeout to accommodate cloud cold starts

/**
 * Custom Error class with HTTP status code and server detail
 */
export class ApiError extends Error {
  constructor(message, status = 500, detail = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Generic fetch wrapper with AbortController timeout handling
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  if (!API_BASE_URL && !import.meta.env.DEV) {
    throw new ApiError(
      "Backend URL (VITE_API_URL) is not configured in Vercel settings. Please add VITE_API_URL in Vercel Environment Variables and redeploy.",
      503
    );
  }

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === "AbortError") {
      throw new ApiError(
        "Request timed out. The cloud server may still be waking up from sleep. Please try again in 10-20 seconds.",
        408
      );
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      error.message?.includes("Failed to fetch")
        ? "Unable to connect to the backend server. Please verify the cloud service is awake and VITE_API_URL is correct."
        : `Connection error: ${error.message || "Failed to reach server"}`,
      503
    );
  }
}

/**
 * Parse JSON safely and throw formatted ApiError on failure
 */
async function parseResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      `Server responded with non-JSON format (Status ${response.status}). If running on a free cloud tier, the service may be warming up.`,
      response.status
    );
  }

  if (!response.ok) {
    const errorMessage = data?.error || data?.detail || `HTTP Error ${response.status}`;
    throw new ApiError(errorMessage, response.status, data);
  }

  return data;
}

export const api = {
  /**
   * Check backend health and model loading status
   * GET /health
   */
  async checkHealth() {
    if (!API_BASE_URL && !import.meta.env.DEV) {
      return {
        status: "unconfigured",
        model_loaded: false,
        error: "VITE_API_URL environment variable is missing on Vercel.",
      };
    }

    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {
        method: "GET",
      }, 15000);
      return await parseResponse(response);
    } catch (error) {
      return {
        status: error.status === 408 ? "waking" : "offline",
        model_loaded: false,
        error: error.message,
      };
    }
  },

  /**
   * Fetch model architecture parameters and test metrics
   * GET /model-info
   */
  async getModelInfo() {
    const response = await fetchWithTimeout(`${API_BASE_URL}/model-info`, {
      method: "GET",
    });
    return await parseResponse(response);
  },

  /**
   * Send MRI image to backend for classification
   * POST /predict
   * @param {File|Blob} fileOrBlob
   * @param {string} filename
   */
  async predictImage(fileOrBlob, filename = "mri_scan.jpg") {
    const formData = new FormData();
    formData.append("file", fileOrBlob, filename);

    const response = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    return await parseResponse(response);
  },

  /**
   * Request Grad-CAM activation visualization and explanation
   * POST /explain
   * @param {File|Blob} fileOrBlob
   * @param {string} filename
   */
  async explainImage(fileOrBlob, filename = "mri_scan.jpg") {
    const formData = new FormData();
    formData.append("file", fileOrBlob, filename);

    const response = await fetchWithTimeout(`${API_BASE_URL}/explain`, {
      method: "POST",
      body: formData,
    });
    return await parseResponse(response);
  },
};
