import React from "react";

export default function StatCard({ value, label, detail, highlight = false }) {
  return (
    <div className={`stat-card ${highlight ? "highlight" : ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}
