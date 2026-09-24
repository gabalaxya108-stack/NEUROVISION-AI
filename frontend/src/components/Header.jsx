import React from "react";

export default function Header({
  activePage,
  setActivePage,
  backendStatus,
  onToggleSidebar,
  onOpenAbout,
}) {
  const isLoaded = backendStatus?.model_loaded === true;
  const isOnline = backendStatus?.status === "ok";
  const isWaking = backendStatus?.status === "waking";
  const isUnconfigured = backendStatus?.status === "unconfigured";

  const pageTitles = {
    dashboard: "Platform Overview",
    analyze: "Analyze Brain MRI",
    performance: "Model Performance & Benchmark",
    "error-analysis": "Comprehensive Error Analysis",
    methodology: "System Methodology & Research Pipeline",
  };

  return (
    <header className="app-header">
      <div className="header-left">
        {/* Mobile Hamburger Toggle */}
        <button
          className="mobile-sidebar-toggle"
          onClick={onToggleSidebar}
          aria-label="Toggle navigation menu"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>

        {/* Breadcrumb & Page Title */}
        <div className="header-page-title-box">
          <div className="header-breadcrumbs">
            <span
              style={{ cursor: "pointer", transition: "color var(--transition-fast)" }}
              onClick={() => setActivePage("dashboard")}
              title="Return to Dashboard"
            >
              NeuroVision AI
            </span>
            <span className="crumb-sep">/</span>
            <span className="crumb-active">{pageTitles[activePage] || "Dashboard"}</span>
          </div>
          <h2 className="header-page-heading">{pageTitles[activePage] || "Dashboard"}</h2>
        </div>
      </div>

      <div className="header-right">
        {/* Model Status Pill */}
        <div className="header-status-pill">
          <span className={`pill-dot ${isOnline && isLoaded ? "dot-online" : isWaking ? "dot-waking" : "dot-offline"}`} />
          <span className="pill-text">
            {isOnline && isLoaded
              ? "Model Ready · 92.81% Acc"
              : isWaking
              ? "Cloud Backend Waking Up..."
              : isUnconfigured
              ? "Backend URL Not Set (Add VITE_API_URL)"
              : isOnline
              ? "Loading Checkpoint..."
              : "Backend Disconnected"}
          </span>
        </div>

        {/* Quick Action: New Scan (if not on analyze page) */}
        {activePage !== "analyze" && (
          <button
            className="btn btn-primary btn-sm header-action-btn"
            onClick={() => setActivePage("analyze")}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Analyze MRI
          </button>
        )}

        {/* About / Citation Modal Trigger */}
        <button
          className="btn btn-secondary btn-sm header-about-btn"
          onClick={onOpenAbout}
          title="About Platform & Research Disclaimers"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <span className="header-about-text">About</span>
        </button>
      </div>
    </header>
  );
}
