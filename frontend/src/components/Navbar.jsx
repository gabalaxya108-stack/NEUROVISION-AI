import React, { useState } from "react";

export default function Navbar({ activePage, setActivePage, backendStatus }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: "dashboard", label: "Dashboard" },
    { id: "analyze", label: "Analyze MRI" },
    { id: "performance", label: "Model Performance" },
    { id: "error-analysis", label: "Error Analysis" },
    { id: "methodology", label: "Methodology" },
  ];

  const handleNavClick = (id) => {
    setActivePage(id);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const isOnline = backendStatus?.status === "ok" && backendStatus?.model_loaded;

  return (
    <header className="navbar">
      <div className="container navbar-container">
        {/* Brand */}
        <a
          href="#dashboard"
          className="navbar-brand"
          onClick={(e) => {
            e.preventDefault();
            handleNavClick("dashboard");
          }}
        >
          <div className="brand-icon">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2a5 5 0 0 1 5 5v1a4 4 0 0 1 4 4 4 4 0 0 1-4 4v1a5 5 0 0 1-5 5 5 5 0 0 1-5-5v-1a4 4 0 0 1-4-4 4 4 0 0 1 4-4V7a5 5 0 0 1 5-5z" />
              <path d="M12 2v20" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-title">BRAIN MRI AI</span>
            <span className="brand-subtitle">Research Classifier</span>
          </div>
        </a>

        {/* Backend Status Pill */}
        <div
          className="backend-status-pill"
          title={isOnline ? "Backend API connected & Model resident in memory" : "Backend offline or unreachable"}
        >
          <span className={`status-dot ${isOnline ? "online" : "offline"}`} />
          <span>{isOnline ? "API Online" : "API Offline"}</span>
        </div>

        {/* Desktop Navigation Links */}
        <nav>
          <ul className={`navbar-links ${mobileMenuOpen ? "open" : ""}`}>
            {navItems.map((item) => (
              <li key={item.id}>
                <button
                  className={`nav-link ${activePage === item.id ? "active" : ""}`}
                  onClick={() => handleNavClick(item.id)}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        {/* Mobile Toggle Button */}
        <button
          className="mobile-toggle"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle navigation menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileMenuOpen ? (
              <path d="M18 6L6 18M6 6l12 12" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>
    </header>
  );
}
