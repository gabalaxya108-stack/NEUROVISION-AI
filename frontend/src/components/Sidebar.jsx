import React from "react";

export default function Sidebar({
  activePage,
  setActivePage,
  backendStatus,
  isOpen,
  setIsOpen,
  onOpenAbout,
}) {
  const isLoaded = backendStatus?.model_loaded === true;
  const isOnline = backendStatus?.status === "ok";

  const navItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="7" height="9" rx="1" />
          <rect x="14" y="3" width="7" height="5" rx="1" />
          <rect x="14" y="12" width="7" height="9" rx="1" />
          <rect x="3" y="16" width="7" height="5" rx="1" />
        </svg>
      ),
      badge: null,
    },
    {
      id: "analyze",
      label: "Analyze MRI",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
          <line x1="12" y1="22.08" x2="12" y2="12" />
        </svg>
      ),
      badge: "LIVE",
    },
    {
      id: "performance",
      label: "Model Performance",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      ),
      badge: "92.8%",
    },
    {
      id: "error-analysis",
      label: "Error Analysis",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      ),
      badge: null,
    },
    {
      id: "methodology",
      label: "Methodology",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="12 2 2 7 12 12 22 7 12 2" />
          <polyline points="2 17 12 22 22 17" />
          <polyline points="2 12 12 17 22 12" />
        </svg>
      ),
      badge: null,
    },
  ];

  const handleNavClick = (id) => {
    setActivePage(id);
    if (setIsOpen) setIsOpen(false); // Close mobile drawer
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setIsOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className={`app-sidebar ${isOpen ? "open" : ""}`}>
        {/* Brand Logo & Title */}
        <div className="sidebar-brand-container">
          <div className="sidebar-brand" onClick={() => handleNavClick("dashboard")}>
            <div className="brand-logo-icon">
              <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="8" fill="#0f172a" />
                <circle cx="16" cy="16" r="10" stroke="#0284c7" strokeWidth="1.75" strokeDasharray="3 2" />
                <path
                  d="M10 16c0-3.3 2.7-6 6-6s6 2.7 6 6-2.7 6-6 6"
                  stroke="#38bdf8"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <circle cx="16" cy="16" r="2.25" fill="#38bdf8" />
                <circle cx="16" cy="10" r="1.5" fill="#0ea5e9" />
                <circle cx="16" cy="22" r="1.5" fill="#0ea5e9" />
                <circle cx="10" cy="16" r="1.5" fill="#0ea5e9" />
                <circle cx="22" cy="16" r="1.5" fill="#0ea5e9" />
              </svg>
            </div>
            <div className="brand-text-col">
              <span className="brand-title">NeuroVision AI</span>
              <span className="brand-subtitle">Brain MRI Research</span>
            </div>
          </div>

          {/* Close button for mobile */}
          <button
            className="sidebar-close-btn"
            onClick={() => setIsOpen(false)}
            aria-label="Close menu"
          >
            ✕
          </button>
        </div>

        {/* Primary Navigation */}
        <nav className="sidebar-nav">
          <div className="sidebar-section-label">Navigation</div>
          <ul className="sidebar-nav-list">
            {navItems.map((item) => {
              const isActive = activePage === item.id;
              return (
                <li key={item.id}>
                  <button
                    className={`sidebar-nav-item ${isActive ? "active" : ""}`}
                    onClick={() => handleNavClick(item.id)}
                  >
                    <span className="sidebar-nav-icon">{item.icon}</span>
                    <span className="sidebar-nav-label">{item.label}</span>
                    {item.badge && (
                      <span className={`sidebar-nav-badge ${item.id === "analyze" ? "badge-live" : ""}`}>
                        {item.badge}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>

          <div className="sidebar-section-label" style={{ marginTop: "1.5rem" }}>
            Platform & Research
          </div>
          <ul className="sidebar-nav-list">
            <li>
              <button
                className="sidebar-nav-item"
                onClick={() => {
                  onOpenAbout();
                  if (setIsOpen) setIsOpen(false);
                }}
              >
                <span className="sidebar-nav-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                </span>
                <span className="sidebar-nav-label">About / Research</span>
              </button>
            </li>
          </ul>
        </nav>

        {/* Bottom System & Model Status Panel */}
        <div className="sidebar-status-box">
          <div className="status-header">
            <div className="status-indicator-dot">
              <span className={`dot-pulse ${isOnline && isLoaded ? "pulse-green" : "pulse-amber"}`} />
              <span className={`dot-core ${isOnline && isLoaded ? "core-green" : "core-amber"}`} />
            </div>
            <span className="status-title">
              {isOnline && isLoaded ? "Model Ready" : isOnline ? "Loading Checkpoint" : "Offline"}
            </span>
          </div>

          <div className="status-details">
            <div className="status-row">
              <span className="status-key">Backbone:</span>
              <span className="status-val font-mono">EfficientNetB0</span>
            </div>
            <div className="status-row">
              <span className="status-key">Benchmark:</span>
              <span className="status-val font-mono">92.81% Acc</span>
            </div>
            <div className="status-row">
              <span className="status-key">Guardrail:</span>
              <span className="status-val text-success">Active ✓</span>
            </div>
          </div>

          <div className="sidebar-privacy-pill">
            <span>🔒 In-Memory Processing Only</span>
          </div>
        </div>
      </aside>
    </>
  );
}
