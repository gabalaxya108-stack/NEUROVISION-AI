import React from "react";
import { DISCLAIMER_TEXT } from "../utils/constants";

export default function DisclaimerBanner() {
  return (
    <aside className="disclaimer-banner" role="complementary" aria-label="Research and educational disclaimer">
      <div className="container disclaimer-content">
        <span className="disclaimer-icon" aria-hidden="true">
          ⚠️
        </span>
        <span>
          <strong>Research Notice:</strong> {DISCLAIMER_TEXT}
        </span>
      </div>
    </aside>
  );
}
