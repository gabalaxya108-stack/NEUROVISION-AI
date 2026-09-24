import React from "react";
import { DISCLAIMER_TEXT } from "../utils/constants";

export default function Footer({ setActivePage, onOpenAbout }) {
  return (
    <footer className="app-footer">
      <div className="container footer-inner">
        <div className="footer-top">
          <div className="footer-brand">
            <h4 className="footer-title">NeuroVision AI — Research Platform</h4>
            <p className="footer-desc">
              Educational deep learning platform for four-class brain MRI scan classification using EfficientNetB0
              with visual Grad-CAM feature attribution and error calibration.
            </p>
          </div>

          <div className="footer-nav">
            <button className="footer-link" onClick={() => setActivePage("dashboard")}>
              Overview
            </button>
            <button className="footer-link" onClick={() => setActivePage("analyze")}>
              Analyze MRI
            </button>
            <button className="footer-link" onClick={() => setActivePage("performance")}>
              Benchmark
            </button>
            <button className="footer-link" onClick={() => setActivePage("error-analysis")}>
              Error Analysis
            </button>
            <button className="footer-link" onClick={() => setActivePage("methodology")}>
              Methodology
            </button>
            {onOpenAbout && (
              <button className="footer-link" onClick={onOpenAbout}>
                About
              </button>
            )}
          </div>
        </div>

        {/* Academic & Educational Notice Card */}
        <div className="footer-disclaimer-card">
          <strong>Academic & Research Notice:</strong> {DISCLAIMER_TEXT} Model confidence scores and Grad-CAM
          activation heatmaps represent computational feature correlations and must not be used as clinical findings
          or radiological diagnoses.
        </div>

        <div className="footer-bottom">
          <span>NeuroVision AI &bull; Brain MRI Classification Research Initiative &bull; 2026</span>
          <span>EfficientNetB0 &bull; 92.81% Test Accuracy (N=1,600) &bull; Held-Out Test Evaluated</span>
        </div>
      </div>
    </footer>
  );
}
