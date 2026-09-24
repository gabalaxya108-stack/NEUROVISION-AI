import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Footer from "./components/Footer";
import DisclaimerBanner from "./components/DisclaimerBanner";
import AboutModal from "./components/AboutModal";

import Dashboard from "./pages/Dashboard";
import AnalyzeMRI from "./pages/AnalyzeMRI";
import ModelPerformance from "./pages/ModelPerformance";
import ErrorAnalysis from "./pages/ErrorAnalysis";
import Methodology from "./pages/Methodology";

import { api } from "./services/api";

import "./styles/index.css";
import "./styles/layout.css";
import "./styles/components.css";
import "./styles/pages.css";

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [backendStatus, setBackendStatus] = useState({ status: "checking", model_loaded: false });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [aboutModalOpen, setAboutModalOpen] = useState(false);

  // Periodically check backend health
  useEffect(() => {
    let isMounted = true;

    const checkBackend = async () => {
      try {
        const result = await api.checkHealth();
        if (isMounted) setBackendStatus(result);
      } catch {
        if (isMounted) setBackendStatus({ status: "offline", model_loaded: false });
      }
    };

    checkBackend();
    const intervalId = setInterval(checkBackend, 15000); // Check every 15s

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  // Scroll to top on page change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [activePage]);

  const renderActivePage = () => {
    switch (activePage) {
      case "dashboard":
        return <Dashboard setActivePage={setActivePage} onOpenAbout={() => setAboutModalOpen(true)} />;
      case "analyze":
        return <AnalyzeMRI />;
      case "performance":
        return <ModelPerformance />;
      case "error-analysis":
        return <ErrorAnalysis />;
      case "methodology":
        return <Methodology />;
      default:
        return <Dashboard setActivePage={setActivePage} onOpenAbout={() => setAboutModalOpen(true)} />;
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        backendStatus={backendStatus}
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
        onOpenAbout={() => setAboutModalOpen(true)}
      />

      <div className="app-main-layout">
        <DisclaimerBanner />
        <Header
          activePage={activePage}
          setActivePage={setActivePage}
          backendStatus={backendStatus}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          onOpenAbout={() => setAboutModalOpen(true)}
        />
        <main className="app-content-area">
          <div key={activePage} className="page-enter">
            {renderActivePage()}
          </div>
        </main>
        <Footer
          setActivePage={setActivePage}
          onOpenAbout={() => setAboutModalOpen(true)}
        />
      </div>

      <AboutModal
        isOpen={aboutModalOpen}
        onClose={() => setAboutModalOpen(false)}
      />
    </div>
  );
}
