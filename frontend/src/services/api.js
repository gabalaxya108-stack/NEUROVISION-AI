/**
 * Centralized API Client for Brain MRI Classification Backend.
 * Communicates with FastAPI endpoints (/health, /model-info, /predict, /explain).
 * Handles network failures, abort timeouts, and structured error responses.
 */

import { API_BASE_URL } from "../utils/constants";

const DEFAULT_TIMEOUT_MS = 30000; // 30 seconds timeout for GPU/CPU inference

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
      throw new ApiError("Request timed out. The backend server took too long to respond.", 408);
    }
    throw new ApiError(
      "Unable to connect to the analysis service. Please try again.",
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
    throw new ApiError(`Server responded with non-JSON format (Status ${response.status})`, response.status);
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
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {
        method: "GET",
      }, 5000);
      return await parseResponse(response);
    } catch (error) {
      return {
        status: "offline",
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
